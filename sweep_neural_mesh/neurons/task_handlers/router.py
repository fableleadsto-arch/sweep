"""
Task Router — classifies queries and dispatches to the correct handler.

The router performs lightweight pattern matching to determine the query
category, then dispatches to the appropriate handler.  Each handler
returns a typed result that the router wraps in a TaskClassification.

Categories:
  - logic:     deduction, induction, syllogisms, analogies, boolean, set theory
  - math:      arithmetic, equations, word problems, verification, number theory
  - evidence:  corroboration, contradiction, source ranking, entity resolution
  - temporal:  date math, timelines, chronological, day-of-week, age
  - causal:    chain reasoning, effect prediction, root cause, counterfactual
"""
from __future__ import annotations

from services.model_loading import serialized_model_load

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .logic import LogicHandler, LogicResult
from .math import MathHandler, MathResult
from .evidence import EvidenceHandler, EvidenceResult
from .temporal import TemporalHandler, TemporalResult
from .causal import CausalHandler, CausalResult


@dataclass(frozen=True, slots=True)
class TaskClassification:
    """Result of classifying and routing a query."""
    query: str
    category: str
    subcategory: str
    answer: str
    confidence: float
    method: str
    details: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


class TaskRouter:
    """Classifies queries and dispatches to the appropriate handler.

    Usage::

        router = TaskRouter()
        result = router.route("2 + 2 = ?")
        print(result.category, result.answer)

        result = router.route("Why does it rain?", evidence=["Water evaporates..."])
        print(result.category, result.answer)
    """

    def __init__(self) -> None:
        self._logic = LogicHandler()
        self._math = MathHandler()
        self._evidence = EvidenceHandler()
        self._temporal = TemporalHandler()
        self._causal = CausalHandler()

    def route(
        self,
        query: str,
        evidence: list[str] | None = None,
    ) -> TaskClassification:
        """Classify and route a query to the appropriate handler.

        Returns a TaskClassification with the answer, category, and metadata.
        """
        t0 = time.perf_counter()
        q = query.strip()
        ev = evidence or []

        # Classify the query
        category, subcategory = self._classify(q)

        # Build handler list: primary + fallbacks
        handlers = []
        if category == "logic":
            handlers = [self._logic, self._evidence, self._causal]
        elif category == "math":
            handlers = [self._math, self._logic]
        elif category == "evidence":
            handlers = [self._evidence, self._logic]
        elif category == "temporal":
            handlers = [self._temporal, self._causal]
        elif category == "causal":
            handlers = [self._causal, self._temporal, self._evidence]
        else:
            # Unknown: try all handlers except evidence. The evidence
            # handler's corroboration path answers ANY query that comes with
            # 2+ similar evidence items, so it must only run for queries
            # explicitly classified as evidence tasks (corroborate/verify/
            # contradict/...). Otherwise generic questions with evidence are
            # answered from one corroboration cluster instead of the full
            # reasoning pipeline.
            handlers = [self._logic, self._math, self._temporal, self._causal]

        # Check evidence relevance before processing
        # If evidence is provided but clearly irrelevant to the query,
        # reduce confidence to avoid answering from knowledge alone
        ev_relevant = True
        if ev:
            q_words = set(re.findall(r'\b[a-z]{3,}\b', q.lower()))
            ev_text = ' '.join(ev).lower()
            ev_words = set(re.findall(r'\b[a-z]{3,}\b', ev_text))
            overlap = q_words & ev_words
            if len(overlap) == 0 and len(q_words) > 3:
                ev_relevant = False

        for handler in handlers:
            result = handler.process(q, ev)
            if result.confidence >= 0.5 and result.answer:
                # If evidence is irrelevant, skip this handler
                # (don't answer from knowledge alone when evidence doesn't support it)
                if not ev_relevant:
                    continue
                # Determine the actual category from the handler type
                if isinstance(result, LogicResult):
                    actual_cat = "logic"
                    details = {"reasoning_chain": result.reasoning_chain, **result.metadata}
                elif isinstance(result, MathResult):
                    actual_cat = "math"
                    details = {"steps": result.steps, **result.metadata}
                elif isinstance(result, EvidenceResult):
                    actual_cat = "evidence"
                    details = result.details
                elif isinstance(result, TemporalResult):
                    actual_cat = "temporal"
                    details = {"steps": result.steps, **result.metadata}
                elif isinstance(result, CausalResult):
                    actual_cat = "causal"
                    details = {"chain": result.chain, **result.metadata}
                else:
                    continue

                return TaskClassification(
                    query=q, category=actual_cat, subcategory=subcategory,
                    answer=result.answer, confidence=result.confidence,
                    method=result.method, details=details,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

        # No handler produced a confident result
        return TaskClassification(
            query=q, category=category, subcategory=subcategory,
            answer="", confidence=0.0, method="none",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def _classify(self, q: str) -> tuple[str, str]:
        """Classify a query into a category and subcategory."""
        q_lower = q.lower().strip()

        # ── Logic ─────────────────────────────────────────
        if re.search(r"\ball\s+\w+\s+are\b", q_lower):
            return "logic", "syllogism"
        if re.search(r"\bno\s+\w+\s+are\b", q_lower):
            return "logic", "syllogism"
        if re.search(r"\bif\s+.+\s+then\b", q_lower) or re.search(r"\bif\s+\w+.+,\s+\w+", q_lower):
            return "logic", "deduction"
        if re.search(r"\b(true|false)\s+(and|or|xor|nand|nor)\b", q_lower):
            return "logic", "boolean"
        if re.search(r"\bnot\s+(true|false)\b", q_lower):
            return "logic", "boolean"
        if re.search(r"\bunion\s+of\b", q_lower):
            return "logic", "set_theory"
        if re.search(r"\bintersection\s+of\b", q_lower):
            return "logic", "set_theory"
        if re.search(r"\bsubset\s+of\b", q_lower):
            return "logic", "set_theory"
        if re.search(r"\bis\s+to\s+.+\s+as\s+.+\s+is\s+to\b", q_lower):
            return "logic", "analogy"
        # Min/max BEFORE the comma-sequence pattern below, otherwise
        # "largest of 100, 200 and 300" is misread as a number sequence
        # and answered with the (wrong) next term.
        if re.search(r"\b(largest|greatest|biggest|smallest|least|lowest|maximum|minimum|highest)\s+of\b", q_lower):
            return "math", "minmax"
        if re.search(r"\bremainder\s+of\b", q_lower):
            return "math", "arithmetic"
        if re.search(r"\bhow\s+many\s+elements?\s+in\s+the\s+(union|intersection)\b", q_lower):
            return "logic", "set_theory"
        if re.search(r"\bpattern\b|\bsequence\b|\bwhat\s+comes\s+next\b", q_lower):
            return "logic", "induction"
        if re.search(r"\d+\s*[,\s]\s*\d+\s*[,\s]\s*\d+\s*[,\s]\s*\d+\s*[,\s]*\s*\?", q_lower):
            return "logic", "induction"
        if re.search(r"\bis\s+.+\s+(a|an)\s+\w+\??$", q_lower):
            return "logic", "classification"

        # ── Math ──────────────────────────────────────────
        if re.search(r"\b(is|what)\s+\d+\s*(plus|minus|times|multiplied|divided|over|\+|\-|\*|/)\s*\d+", q_lower):
            return "math", "arithmetic"
        if re.search(r"\d+\s*(plus|minus|times|multiplied|divided|over|\+|\-|\*|/)\s*\d+", q_lower):
            return "math", "arithmetic"
        if re.search(r"\bsqrt\b|\bsquare\s+root\b|\bfactorial\b|\d+!", q_lower):
            return "math", "arithmetic"
        if re.search(r"\bx\s*[\*×]?\s*x?\s*[+\-]\s*\d+\s*x?\s*=\s*\d+", q_lower):
            return "math", "linear_equation"
        if re.search(r"\bx\s*[\^²]\s*2", q_lower):
            return "math", "quadratic"
        if re.search(r"\bis\s+\d+\s+prime\b", q_lower):
            return "math", "number_theory"
        if re.search(r"\bfactors?\s+(of\s+)?\d+", q_lower):
            return "math", "number_theory"
        if re.search(r"\b(gcd|lcm)\s+of\b", q_lower):
            return "math", "number_theory"
        if re.search(r"\b\w+\s+prime\b", q_lower):
            return "math", "number_theory"
        if re.search(r"\bpercent|%|\bpercentage\b", q_lower):
            return "math", "percentage"
        if re.search(r"\bconvert\b.*\bto\b", q_lower):
            return "math", "unit_conversion"
        if re.search(r"\bverify\b|\bcheck\b.*\bcorrect\b|\bis\s+.+\s*=\s*.+\s*(correct|right|true)", q_lower):
            return "math", "verification"
        if re.search(r"\b(add|subtract|multiply|divide)\b.*\b(how\s+many|total|result)", q_lower):
            return "math", "word_problem"
        if re.search(r"\b\d+\s+items?\s+cost\b", q_lower):
            return "math", "word_problem"

        # ── Temporal ──────────────────────────────────────
        if re.search(r"\bwhen\s+(did|was|has|is)\b", q_lower):
            return "temporal", "event_lookup"
        if re.search(r"\bwhat\s+year\b", q_lower):
            return "temporal", "event_lookup"
        if re.search(r"\bhow\s+(many\s+)?days?\s+between\b", q_lower):
            return "temporal", "date_difference"
        if re.search(r"\bhow\s+long\s+between\b", q_lower):
            return "temporal", "duration"
        if re.search(r"\bdays?\s+(after|before|from)\b", q_lower):
            return "temporal", "date_math"
        if re.search(r"\btimeline\b|\bchronological\b|\border\s+(by|the)\s+date\b", q_lower):
            return "temporal", "timeline"
        if re.search(r"\bwhat\s+(happened|occurred)\s+(first|last|before|after)\b", q_lower):
            return "temporal", "chronological"
        if re.search(r"\bwhat\s+(happened|was)\s+first\b", q_lower):
            return "temporal", "chronological"
        if re.search(r"\bhow\s+old\b|\bwhat\s+is.*\bage\b", q_lower):
            return "temporal", "age"
        if re.search(r"\bwhat\s+day\b", q_lower):
            return "temporal", "day_of_week"

        # ── Causal ────────────────────────────────────────
        if re.search(r"\bwhy\s+(does?|do|is|are|did|has|can|would)\b", q_lower):
            return "causal", "chain"
        if re.search(r"\bwhat\s+(causes?|makes?|leads?\s+to)\b", q_lower):
            return "causal", "chain"
        if re.search(r"\broot\s+cause\b", q_lower):
            return "causal", "root_cause"
        if re.search(r"\bwhat\s+if\b", q_lower):
            return "causal", "counterfactual"
        if re.search(r"\bwhat\s+(happens?|would\s+happen)\s+(if|when)\b", q_lower):
            return "causal", "effect_prediction"

        # ── Evidence ──────────────────────────────────────
        if re.search(r"\b(corroborate|confirm|verify|support)\b.*\bevidence\b", q_lower):
            return "evidence", "corroboration"
        if re.search(r"\bcontradict|conflict|disagree\b", q_lower):
            return "evidence", "contradiction"
        if re.search(r"\b(source|reliable|trust|credible|rank)\b", q_lower):
            return "evidence", "source_ranking"
        if re.search(r"\b(same|different|entity|person|match|duplicate)\b", q_lower):
            return "evidence", "entity_resolution"
        if re.search(r"\b(verify|check|true|correct|supported|claim)\b", q_lower):
            return "evidence", "claim_verification"
        if re.search(r"\b(score|relevance|strength|quality|rate)\b", q_lower):
            return "evidence", "evidence_scoring"

        # ── Neural zero-shot fallback ──────────────────────────────
        # Pattern rules above are deterministic and tried first. For natural
        # language that matches none of them, use the pretrained MNLI model
        # (cross-encoder/nli-deberta-v3-base, already cached) to pick the
        # closest category from hypothesis descriptions — no task-specific
        # training needed.
        zs = _zero_shot_category(q)
        if zs is not None:
            return zs

        return "unknown", "none"

    def _zero_shot_ok(self, q: str) -> bool:
        """Guard so the zero-shot fallback only fires on genuine questions."""
        return q.rstrip().endswith(("?", ".")) or len(q.split()) > 3


# ────────────────────────────────────────────────────────────────────
# Zero-shot classifier (pretrained MNLI — no Sweep-specific training)
# ────────────────────────────────────────────────────────────────────

_ZS_LABELS: dict[str, str] = {
    # hypothesis text -> category (subcategory resolved by handler rules)
    "This is a question about a date, a year, a timeline, a schedule, a duration, or how much time passed between events.": "temporal",
    "This is an arithmetic calculation, a math word problem, or a question about numbers.": "math",
    "This is a question about causes and effects, about why something happens or what leads to what.": "causal",
    "This is a question about logical deduction, what must be true, or a syllogism.": "logic",
}
_zs_lock = threading.Lock()  # guards inference; created eagerly (a None
                             # lock here made `with _zs_lock:` raise and the
                             # caller silently swallow every classification)
_zs_model = None
_zs_tok = None
_zs_ready = False
_zs_init_lock = threading.Lock()


def _zero_shot_category(query: str) -> tuple[str, str] | None:
    """Classify via entailment between the query and category hypotheses.

    Returns (category, "zero_shot") or None if the model isn't ready or no
    hypothesis clears the threshold. Runs in <50ms once loaded; loads lazily
    so the first call doesn't block routing.

    Calibration note: the entailment score separates *domain* but the raw
    cross-encoder over-entails "math" for nearly any declarative sentence
    (stories, chat). A query is only routed when it is a genuine question
    (interrogative or explicit task verb) — chit-chat falls through to the
    cortex's chat pipeline instead.
    """
    global _zs_lock, _zs_model, _zs_tok, _zs_ready
    import threading
    import re as _re
    if not _re.search(r"\b(what|when|where|who|why|how|which|is|are|was|were|does|do|did|can|calculate|compute|convert|solve|verify|check|if)\b",
                      query.lower()):
        return None
    # The zero-shot classifier needs the RAW MNLI head: the engine's
    # deployed artifact is pair-finetuned (supports/refutes), and its head
    # reads unrelated-domain pairs as neutral, so it can't score category
    # hypotheses. Load the raw cross-encoder lazily once (dedicated ~280MB
    # with fp32; runs <50ms per batch of 4 hypotheses).
    if not _zs_ready:
        if not getattr(_zs_load_worker, "_started", False):
            _zs_load_worker._started = True
            threading.Thread(target=_zs_load_worker, daemon=True).start()
        if not _zs_ready:
            return None
    try:
        import torch
        hypotheses = list(_ZS_LABELS.keys())
        pairs = [(query, h) for h in hypotheses]
        with _zs_lock:
            enc = _zs_tok(pairs, return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                logits = _zs_model(**enc).logits
        # Raw MNLI head: id2label = {0: contradiction, 1: entailment, 2: neutral}
        # -> entailment probability is column 1.
        probs = torch.softmax(logits, dim=-1)[:, 1]
        best = int(probs.argmax())
        if float(probs[best]) < 0.55:
            return None
        cat = _ZS_LABELS[hypotheses[best]]
        return cat, "zero_shot"
    except Exception:
        return None


@serialized_model_load
def _zs_load_worker() -> None:
    global _zs_model, _zs_tok, _zs_ready
    try:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        _zs_tok = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-base")
        _zs_model = AutoModelForSequenceClassification.from_pretrained(
            "cross-encoder/nli-deberta-v3-base")
        _zs_model.eval()
        _zs_ready = True  # raw MNLI head: id2label = contradiction/entailment/neutral
    except Exception:
        _zs_ready = False
