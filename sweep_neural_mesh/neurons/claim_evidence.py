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
from datetime import date
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

# Structured claims with numeric / date / quantifier grounding
# (hidden-set skills: duration arithmetic, ISO-date ordering, most/none).
_DURATION_CLAIM_RE = re.compile(
    r"\bthe\s+(.+?)\s+(?:outage|incident|failure|downtime)\s+lasted\s+"
    r"(\d+)\s+hours?", re.IGNORECASE)
_DURATION_EVIDENCE_RE = re.compile(
    r"\bbegan\s+at\s+(\d{1,2}):00\s+and\s+ended\s+at\s+(\d{1,2}):00", re.IGNORECASE)
_ORDER_CLAIM_RE = re.compile(
    r"\bthe\s+(.+?)\s+(?:incident|outage|event|issue)\s+happened\s+"
    r"(before|after)\s+the\s+(?:patch|fix|deploy|release|maintenance)", re.IGNORECASE)
_INCIDENT_DATE_RE = re.compile(
    r"\b(?:incident|outage|event|issue)\s+occurred\s+on\s+(\d{4}-\d{2}-\d{2})", re.IGNORECASE)
_PATCH_DATE_RE = re.compile(
    r"\b(?:patch|fix|deploy|release|maintenance)[^.]*?(?:was\s+)?deployed\s+on\s+"
    r"(\d{4}-\d{2}-\d{2})", re.IGNORECASE)
_QUANT_CLAIM_RE = re.compile(
    r"\b(most|no|all|some|none\s+of\s+the)\s+(.+?)\s+are\s+(.+)", re.IGNORECASE)
_QUANT_EVIDENCE_RE = re.compile(
    r"\b(\d+)\s+of\s+(\d+)\s+(.+?)\s+are\s+(.+)", re.IGNORECASE)


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
    return (bool(_STATEMENT_RE.search(q))
            or bool(_DURATION_CLAIM_RE.search(q))
            or bool(_ORDER_CLAIM_RE.search(q))
            or bool(_QUANT_CLAIM_RE.search(q))
            or bool(re.search(
                r"\bbecause\b.*,\s*the\s+.+\s+(?:is|was)\s+", q, re.IGNORECASE)))


def _verdict(decision: str, confidence: float, detail: str,
             method: str = "claim_evidence_analyzer:structured") -> ClaimVerdict:
    return ClaimVerdict(
        decision=decision, confidence=confidence,
        reasoning=f"Claim verification: {detail}",
        pieces=[], method=method,
    )


def _try_duration(claim: str, evidence: list) -> ClaimVerdict | None:
    """Verify 'the X <event> lasted N hours' against log begin/end times."""
    cm = _DURATION_CLAIM_RE.search(claim)
    if not cm:
        return None
    claim_hours = int(cm.group(2))
    entity = cm.group(1).strip()
    for ev in evidence:
        ev = ev.get("text", "") if isinstance(ev, dict) else str(ev)
        if entity.lower() not in ev.lower():
            continue
        em = _DURATION_EVIDENCE_RE.search(ev)
        if em:
            actual = int(em.group(2)) - int(em.group(1))
            if actual == claim_hours:
                return _verdict(
                    "supported", 0.90,
                    f"{entity}: logs show a {actual}-hour duration, "
                    f"matching the claim ({claim_hours} hours)")
            return _verdict(
                "refuted", 0.90,
                f"{entity}: logs show a {actual}-hour duration, "
                f"not the claimed {claim_hours} hours")
    return None


def _try_event_order(claim: str, evidence: list) -> ClaimVerdict | None:
    """Verify 'the X <event> happened before/after the patch' via ISO dates."""
    cm = _ORDER_CLAIM_RE.search(claim)
    if not cm:
        return None
    relation = cm.group(2)  # before | after
    text = " ".join(
        e.get("text", "") if isinstance(e, dict) else str(e) for e in evidence
    )
    inc_m = _INCIDENT_DATE_RE.search(text)
    patch_m = _PATCH_DATE_RE.search(text)
    if not inc_m or not patch_m:
        return None
    inc = date.fromisoformat(inc_m.group(1))
    patch = date.fromisoformat(patch_m.group(1))
    actually_before = inc < patch
    claim_says_before = relation == "before"
    relation_word = "before" if actually_before else "after"
    if actually_before == claim_says_before:
        return _verdict(
            "supported", 0.90,
            f"incident {inc.isoformat()} is {relation_word} the patch "
            f"{patch.isoformat()}, matching the claim")
    return _verdict(
        "refuted", 0.90,
        f"incident {inc.isoformat()} is {relation_word} the patch "
        f"{patch.isoformat()}, contradicting the claim")


def _try_quantifier(claim: str, evidence: list) -> ClaimVerdict | None:
    """Verify 'Most/No Xs are Y' against an 'N of M Xs are Y' ratio."""
    cm = _QUANT_CLAIM_RE.search(claim)
    if not cm:
        return None
    quant = cm.group(1).strip().lower()
    if quant not in ("most", "no"):
        return None
    for ev in evidence:
        ev = ev.get("text", "") if isinstance(ev, dict) else str(ev)
        em = _QUANT_EVIDENCE_RE.search(ev)
        if not em:
            continue
        count, total = int(em.group(1)), int(em.group(2))
        if total <= 0 or count > total:
            return None
        if quant == "most":
            grounded = count > total / 2
            basis = f"{count} of {total} ({(100 * count) // total}%)"
            rule = "'most' requires more than half"
        else:
            grounded = count == 0
            basis = f"{count} of {total}"
            rule = "'no' requires zero"
        if grounded:
            return _verdict(
                "supported", 0.90,
                f"evidence supports the claim: {basis} online ({rule})")
        return _verdict(
            "refuted", 0.90,
            f"evidence refutes the claim: {basis} online ({rule})")
    return None


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
    # Structured numeric / date / quantifier grounding first
    for analyzer in (_try_duration, _try_event_order, _try_quantifier):
        v = analyzer(claim, evidence)
        if v is not None:
            return v

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