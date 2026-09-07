"""
Claim-Evidence Analyzer — deterministic verification of a claim against evidence.

The fine-tuned neural evidence classifier shipped degenerate (trained on single
sentences, inferred on sentence pairs), so the cortex could not distinguish
"supports" from "refutes" and never abstained. This module implements a robust,
explainable claim-verification algorithm that:

  * Extracts the entity + predicate from a claim ("The zorp is charged.").
  * Scores every evidence sentence against the claim:
      - asserts the same predicate              -> supports
      - asserts the negation of the predicate   -> refutes
      - double negation ("not not")             -> supports
      - hedged ("might be", "around 1992")      -> neutral (uncertain)
      - different predicate on the same entity  -> neutral (irrelevant)
      - conflicting sources (support + refute)  -> unknown (cannot decide)
  * Returns supported / refuted / unknown with an honest confidence and a
    human-readable reasoning chain.

This mirrors how an evidence pipeline should behave: aggregate per-piece
verdicts, abstain when the evidence is ambiguous, irrelevant, or hedged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ──────────────────────────────────────────────────────────────────────
# Markers
# ──────────────────────────────────────────────────────────────────────

_NEG_PATTERNS = [
    r"\bnot\b", r"\bnever\b", r"\bno\b", r"\bneither\b",
    r"\bisn't\b", r"\baren't\b", r"\bwasn't\b", r"\bweren't\b",
    r"\bdoesn't\b", r"\bdon't\b", r"\bdidn't\b", r"\bcannot\b",
    r"\bcan't\b", r"\bwithout\b",
]

# Sentence-level negation wrappers ("It is false that ...")
_NEG_WRAPPER_PATTERNS = [
    r"\bfalse\s+that\b", r"\bincorrect\s+that\b", r"\bwrong\s+that\b",
    r"\bdenies?\s+that\b", r"\bdisputes?\s+that\b", r"\bcontradicts?\s+that\b",
    r"\bis\s+not\s+true\b", r"\bis\s+false\b",
]

_HEDGE_PATTERNS = [
    r"\bmight\b", r"\bcould\b", r"\bmay\b", r"\bpossibly\b",
    r"\bprobably\b", r"\baround\b", r"\bapproximately\b", r"\broughly\b",
    r"\bsupposedly\b", r"\ballegedly\b", r"\breportedly\b",
    r"\bmaybe\b", r"\blikely\b", r"\bunclear\b", r"\buncertain\b",
]

_COPULA = r"(?:is|are|was|were|has|have|might\s+be|could\s+be|may\s+be|possibly\s+is)"

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "has", "have", "had",
    "and", "or", "but", "in", "on", "at", "to", "of", "for", "with",
    "by", "from", "that", "this", "it", "as", "per", "its", "their",
    "says", "said", "say", "report", "reports", "records", "record",
    "source", "sources", "according", "log", "logs", "diagnostic",
    "currently", "now", "still", "appears", "seems", "being",
}

_STATEMENT_RE = re.compile(
    rf"\bthe\s+(.+?)\s+({_COPULA})\s+(.+)", re.IGNORECASE
)


@dataclass
class EvidenceVerdict:
    """Verdict for a single evidence piece against a claim."""
    verdict: str          # "supports" | "refutes" | "neutral"
    entity: str
    claim_predicate: str
    evidence_predicate: str
    detail: str


@dataclass
class ClaimVerdict:
    """Overall claim-verification result."""
    decision: str         # "supported" | "refuted" | "unknown"
    confidence: float
    reasoning: str
    pieces: list[EvidenceVerdict] = field(default_factory=list)
    method: str = "claim_evidence_analyzer"


def _has_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


def _content_words(text: str) -> set[str]:
    """Lowercased content words, stems collapsed by suffix-stripping."""
    words = set(re.findall(r"[a-z]{3,}", text.lower()))
    words -= _STOP
    return words


def _stems(words: set[str]) -> set[str]:
    out: set[str] = set()
    for w in words:
        s = w
        for suffix in ("ing", "ed", "es", "s"):
            if len(w) > 4 and w.endswith(suffix):
                s = w[: -len(suffix)]
                break
        out.add(s)
    return out


def _predicates_overlap(a: str, b: str) -> bool:
    """True if two predicate phrases share a content-word stem."""
    wa = _stems(_content_words(a))
    wb = _stems(_content_words(b))
    if not wa or not wb:
        return False
    return bool(wa & wb)


def looks_like_claim(query: str) -> bool:
    """True if the query is a declarative claim sentence (not a question)."""
    q = query.strip()
    if not q or "?" in q:
        return False
    if re.search(r"\b(what|who|why|how|which|when|where|compare|is\s+there)\b", q.lower()):
        return False
    return bool(_STATEMENT_RE.search(q)) or bool(
        re.search(r"\bbecause\b.*,\s*the\s+.+\s+(?:is|was)\s+", q, re.IGNORECASE)
    )


def _extract(statement: str) -> dict | None:
    """Extract (entity, predicate, sentence) from a statement about an entity.

    Handles attribution wrappers ("Source A says the zorp is charged."),
    negation ("is not", "not not", "It is false that ..."), and multi-word
    entities ("the quantum relay is calibrated").
    """
    s = statement.strip()
    m = _STATEMENT_RE.search(s)
    if not m:
        return None
    entity = m.group(1).strip()
    copula = m.group(2).strip()
    pred = m.group(3).strip()
    pred_lower = pred.lower()

    # sentence-level negation wrapper flips polarity
    wrapped_neg = _has_any(s.lower(), _NEG_WRAPPER_PATTERNS)

    # double negation cancels out
    if "not not" in pred_lower:
        polarity = 1
    else:
        negated = _has_any(pred_lower, _NEG_PATTERNS)
        polarity = -1 if (negated or wrapped_neg) else 1

    # hedge markers may live in the copula ("might be", "possibly is") or the
    # predicate ("around 1992", "probably true")
    hedged = _has_any(pred_lower, _HEDGE_PATTERNS) or _has_any(copula.lower(), _HEDGE_PATTERNS)

    return {
        "entity": entity,
        "predicate": pred,
        "polarity": polarity,
        "hedged": hedged,
        "sentence": s,
    }


def analyze_claim_evidence(claim: str, evidence: list[str]) -> ClaimVerdict:
    """Verify a claim against a list of evidence sentences."""
    claim_info = _extract(claim)
    if claim_info is None:
        # Claim does not follow the entity-predicate template — no verdict.
        return ClaimVerdict(
            decision="unknown", confidence=0.3,
            reasoning="Claim is not a verifiable entity-predicate statement",
            method="claim_evidence_analyzer:no_template",
        )

    c_entity = claim_info["entity"]
    c_pred = claim_info["predicate"]
    pieces: list[EvidenceVerdict] = []
    supports = refutes = neutrals = 0

    for ev in evidence:
        if isinstance(ev, dict):
            ev = ev.get("text", "")
        ev = str(ev)
        info = _extract(ev)
        if info is None:
            pieces.append(EvidenceVerdict(
                verdict="neutral", entity=c_entity,
                claim_predicate=c_pred, evidence_predicate="",
                detail=f"Evidence not about the claim's entity: {ev[:80]}",
            ))
            neutrals += 1
            continue

        if info["entity"].lower() != c_entity.lower():
            pieces.append(EvidenceVerdict(
                verdict="neutral", entity=c_entity,
                claim_predicate=c_pred, evidence_predicate=info["predicate"],
                detail=f"Evidence concerns '{info['entity']}' not '{c_entity}'",
            ))
            neutrals += 1
            continue

        same_pred = _predicates_overlap(c_pred, info["predicate"])
        if not same_pred:
            pieces.append(EvidenceVerdict(
                verdict="neutral", entity=c_entity,
                claim_predicate=c_pred, evidence_predicate=info["predicate"],
                detail=f"Evidence asserts '{info['predicate']}' — different predicate",
            ))
            neutrals += 1
            continue

        if info["hedged"]:
            pieces.append(EvidenceVerdict(
                verdict="neutral", entity=c_entity,
                claim_predicate=c_pred, evidence_predicate=info["predicate"],
                detail=f"Evidence is hedged ('{info['predicate']}') — not definitive",
            ))
            neutrals += 1
            continue

        if info["polarity"] > 0:
            supports += 1
            pieces.append(EvidenceVerdict(
                verdict="supports", entity=c_entity,
                claim_predicate=c_pred, evidence_predicate=info["predicate"],
                detail=f"Evidence asserts the claim directly: '{ev[:80]}'",
            ))
        else:
            refutes += 1
            pieces.append(EvidenceVerdict(
                verdict="refutes", entity=c_entity,
                claim_predicate=c_pred, evidence_predicate=info["predicate"],
                detail=f"Evidence negates the claim: '{ev[:80]}'",
            ))

    # ── Aggregate ──────────────────────────────────────────────────
    detail_lines = [f"Claim: {claim.strip()}"]

    if supports and refutes:
        decision, conf = "unknown", 0.45
        detail_lines.append(f"Conflicting evidence: {supports} support(s) vs {refutes} refutation(s)")
        method = "claim_evidence_analyzer:conflicting"
    elif refutes:
        decision, conf = "refuted", 0.90
        detail_lines.append(f"Evidence refutes the claim ({refutes} refutation(s))")
        method = "claim_evidence_analyzer:refuted"
    elif supports:
        decision, conf = "supported", 0.90
        detail_lines.append(f"Evidence supports the claim ({supports} confirmation(s))")
        method = "claim_evidence_analyzer:supported"
    else:
        decision, conf = "unknown", 0.45
        if neutrals:
            detail_lines.append(f"Evidence is inconclusive ({neutrals} neutral/irrelevant piece(s))")
        method = "claim_evidence_analyzer:insufficient"

    for p in pieces:
        detail_lines.append(f"  - {p.verdict}: {p.detail}")

    return ClaimVerdict(
        decision=decision,
        confidence=conf,
        reasoning="\n".join(detail_lines),
        pieces=pieces,
        method=method,
    )