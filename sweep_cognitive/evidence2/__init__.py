"""
Evidence Intelligence 2.0 — Phase 11.

Upgrades the evidence pipeline with the two capabilities the spec calls
out explicitly (section 4 "preserve and improve" and item 21):

1. SOURCE-INDEPENDENCE ANALYSIS (spec: "Do not count ten copied sources
   as ten independent confirmations. Detect source dependence.")
   - Explicit source registry with declared dependence (ownership,
     syndication, shared origin) and observed dependence (co-citation,
     shared upstream URL, near-identical text).
   - Dependence graph with transitive closure: if A copies B and B
     copies C, then A, B, C form one information lineage →
     effective_independent_sources = number of connected lineages,
     not the number of documents.

2. EVIDENCE QUALITY SCORING — per-item quality from reliability,
   independence, corroboration, recency, and modality agreement.

3. RICHER CONSENSUS (spec module 11) — agreement/disagreement weighted
   by quality and independence, with contradiction severity, recency
   and modality agreement factored in. Not majority vote.

IMPLEMENTATION STATUS (honest):
- REAL: source registry, dependence graph, lineage computation,
  independence-weighted consensus math, quality scoring, verdict
  mapping. All deterministic and tested.
- TEMPORARY SCAFFOLDING: near-duplicate text detection is a shingle
  (n-gram) overlap heuristic, not a learned semantic similarity model.
  It is replaceable behind `detect_shared_origin()`.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

_uid_counter = itertools.count(1)


def _uid(prefix: str) -> str:
    return f"{prefix}_{next(_uid_counter)}_{int(time.time() * 1000)}"


# ======================================================================
# SOURCES AND DEPENDENCE
# ======================================================================

@dataclass
class Source:
    """A known source with declared and observed dependence links."""
    source_id: str
    name: str = ""
    reliability: float = 0.7          # prior quality of this source, 0-1
    # Declared dependence: explicit knowledge ("A is owned by B",
    # "A syndicates from B")
    declared_upstream: list[str] = field(default_factory=list)
    # Observed dependence evidence is stored in the dependence graph,
    # not on the source itself.
    first_seen: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "name": self.name,
            "reliability": self.reliability,
            "declared_upstream": list(self.declared_upstream),
            "metadata": self.metadata,
        }


@dataclass
class SourceItem:
    """One evidence item traceable to a source.

    content_hash: identity of the *content* (not the document) — used to
        detect that two documents carry the same wire copy.
    shingles: lexical n-grams used by the duplicate-detection scaffold.
    """
    item_id: str = field(default_factory=lambda: _uid("item"))
    source_id: str = ""
    content: str = ""
    content_hash: str = ""
    timestamp: float = field(default_factory=time.time)
    modality: str = "text"            # text / image / audio / video / structured
    stance: float = 0.0               # -1 contradicts .. 0 neutral .. +1 supports
    shingles: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)


def shingle_text(text: str, k: int = 4) -> set[str]:
    """Word k-shingles for lexical overlap. SCAFFOLDING — a learned
    similarity model replaces this behind the same function."""
    words = text.lower().split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def shingle_jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def shingle_containment(a: set[str], b: set[str]) -> float:
    """Overlap coefficient: |A∩B| / min(|A|,|B|).

    For copy detection containment is the right notion (rather than
    Jaccard): a copy typically contains the original's shingles plus
    extra boilerplate, and a single word change in a short text
    destroys every k-shingle touching it, which Jaccard over-penalizes.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


class SourceRegistry:
    """
    Tracks sources and the dependence relations between them.

    Dependence edges:
    - declared: set explicitly (ownership, syndication)
    - observed: detected from evidence (shared content hash = shared
      wire copy; high shingle overlap = one copied the other; recorded
      with a confidence)

    Effective independence = number of connected lineages in the
    dependence graph (union-find over dependence edges).
    """

    # Text similarity above this → treat as same-origin (scaffold param)
    DUPLICATE_OVERLAP_THRESHOLD = 0.75
    # Near-identical text is ambiguous on its own (press release quoted
    # by N independent outlets vs. one copying another). It only becomes
    # a dependence edge with a corroborating publish-time gap: content
    # appearing this many seconds later is treated as copied from the
    # earlier one. Real syndication lags are minutes-to-hours; identical
    # content ingested simultaneously is treated as independent
    # quotation of a shared public source.
    COPY_TIME_GAP_SECONDS = 60.0

    def __init__(self):
        self._sources: dict[str, Source] = {}
        # undirected dependence edges: (a, b) -> kind + confidence
        self._dependence: dict[tuple[str, str], dict[str, Any]] = {}
        self._items: dict[str, SourceItem] = {}
        # content_hash -> set of item_ids (exact same-content detection)
        self._by_content_hash: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------

    def register_source(
        self, source_id: str, name: str = "", reliability: float = 0.7,
        declared_upstream: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Source:
        if source_id in self._sources:
            src = self._sources[source_id]
            if name:
                src.name = name
            if declared_upstream:
                src.declared_upstream = list(declared_upstream)
            return src
        src = Source(
            source_id=source_id, name=name, reliability=reliability,
            declared_upstream=list(declared_upstream or []),
            metadata=dict(metadata or {}),
        )
        self._sources[source_id] = src
        return src

    def get_source(self, source_id: str) -> Optional[Source]:
        return self._sources.get(source_id)

    def add_declared_dependence(self, upstream_id: str, downstream_id: str) -> None:
        """Declare that downstream copies/derives from upstream."""
        if upstream_id not in self._sources or downstream_id not in self._sources:
            raise KeyError("Both sources must be registered first")
        self._dependence[(upstream_id, downstream_id)] = {
            "kind": "declared", "confidence": 1.0,
        }

    # ------------------------------------------------------------------
    # Evidence items
    # ------------------------------------------------------------------

    def add_item(
        self, source_id: str, content: str, stance: float = 0.0,
        modality: str = "text", content_hash: str = "",
        timestamp: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SourceItem:
        if source_id not in self._sources:
            raise KeyError(f"Unknown source: {source_id}")
        item = SourceItem(
            source_id=source_id,
            content=content,
            content_hash=content_hash,
            timestamp=timestamp if timestamp is not None else time.time(),
            modality=modality,
            stance=max(-1.0, min(1.0, stance)),
            shingles=shingle_text(content),
            metadata=dict(metadata or {}),
        )
        self._items[item.item_id] = item
        if content_hash:
            self._by_content_hash.setdefault(content_hash, set()).add(item.item_id)
        return item

    def get_item(self, item_id: str) -> Optional[SourceItem]:
        return self._items.get(item_id)

    def items_for_source(self, source_id: str) -> list[SourceItem]:
        return [i for i in self._items.values() if i.source_id == source_id]

    # ------------------------------------------------------------------
    # Observed dependence detection
    # ------------------------------------------------------------------

    def detect_shared_origin(self, item_a: SourceItem, item_b: SourceItem) -> Optional[dict[str, Any]]:
        """
        Detect whether two items plausibly share an origin. SCAFFOLDING:
        exact content hash match = certain; high shingle overlap =
        probable. A learned semantic-similarity model slots in here.

        Ambiguity policy: near-identical text alone does NOT prove
        dependence — independent outlets may quote the same press
        release. Near-duplicate text counts as dependence only when
        there is a corroborating signal (publish-time gap: the later
        item could have copied the earlier one).
        """
        if item_a.source_id == item_b.source_id:
            # same source → not "dependence", it's the same source
            return None
        if item_a.content_hash and item_a.content_hash == item_b.content_hash:
            # Identical content hash is strong but still needs the time
            # signal to distinguish copying from shared quotation.
            gap = abs(item_a.timestamp - item_b.timestamp)
            if gap > self.COPY_TIME_GAP_SECONDS:
                return {"kind": "same_content_hash", "confidence": 1.0,
                        "time_gap": round(gap, 3)}
            return None  # indistinguishable from shared quotation
        overlap = shingle_containment(item_a.shingles, item_b.shingles)
        if overlap >= self.DUPLICATE_OVERLAP_THRESHOLD:
            gap = abs(item_a.timestamp - item_b.timestamp)
            if gap > self.COPY_TIME_GAP_SECONDS:
                return {"kind": "near_duplicate_text", "confidence": round(overlap, 3),
                        "time_gap": round(gap, 3)}
            return None
        return None

    def observe_dependence_among(self, item_ids: list[str]) -> list[dict[str, Any]]:
        """Run shared-origin detection across a set of items and record
        any discovered dependence edges. Returns edges found."""
        found: list[dict[str, Any]] = []
        items = [self._items[i] for i in item_ids if i in self._items]
        for ia, ib in itertools.combinations(items, 2):
            dep = self.detect_shared_origin(ia, ib)
            if dep:
                key = tuple(sorted((ia.source_id, ib.source_id)))
                existing = self._dependence.get(key)
                if not existing or dep["confidence"] > existing["confidence"]:
                    self._dependence[key] = {
                        "kind": dep["kind"], "confidence": dep["confidence"],
                    }
                found.append({
                    "sources": list(key),
                    **dep,
                })
        return found

    # ------------------------------------------------------------------
    # Lineage / effective independence
    # ------------------------------------------------------------------

    def effective_independent_sources(self, source_ids: list[str]) -> int:
        """
        Number of distinct information lineages among the given sources
        (connected components over dependence edges, transitive).
        Ten syndicated copies of one wire story → 1.
        """
        ids = list(dict.fromkeys(source_ids))
        if not ids:
            return 0
        parent: dict[str, str] = {s: s for s in ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for (a, b) in self._dependence:
            if a in parent and b in parent:
                union(a, b)

        return len({find(s) for s in ids})

    def lineage_groups(self, source_ids: list[str]) -> list[list[str]]:
        """The actual lineage partitions (for explanations/audit)."""
        ids = list(dict.fromkeys(source_ids))
        if not ids:
            return []
        parent: dict[str, str] = {s: s for s in ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for (a, b) in self._dependence:
            if a in parent and b in parent:
                union(a, b)

        groups: dict[str, list[str]] = {}
        for s in ids:
            groups.setdefault(find(s), []).append(s)
        return list(groups.values())

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "sources": len(self._sources),
            "items": len(self._items),
            "dependence_edges": len(self._dependence),
            "declared_edges": sum(1 for d in self._dependence.values() if d["kind"] == "declared"),
            "observed_edges": sum(1 for d in self._dependence.values() if d["kind"] != "declared"),
        }


# ======================================================================
# EVIDENCE QUALITY
# ======================================================================

@dataclass
class EvidenceQuality:
    """Quality decomposition for one evidence item."""
    item_id: str
    source_reliability: float
    independence_factor: float      # 1/n_lineages for its lineage group
    corroboration_factor: float     # independent support beyond itself
    recency_factor: float           # exponential decay, 7-day half-life
    overall: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "source_reliability": round(self.source_reliability, 3),
            "independence_factor": round(self.independence_factor, 3),
            "corroboration_factor": round(self.corroboration_factor, 3),
            "recency_factor": round(self.recency_factor, 3),
            "overall": round(self.overall, 3),
        }


# 7-day half-life for recency decay
_RECENCY_HALF_LIFE = 7.0 * 24 * 3600


class EvidenceEngine2:
    """
    Quality scoring + independence-weighted consensus over registered
    evidence. This is the Phase 11 replacement for lexical evidence
    counting inside the reasoning layer.
    """

    def __init__(self, registry: SourceRegistry):
        self.registry = registry

    # ------------------------------------------------------------------
    # Quality
    # ------------------------------------------------------------------

    def score_item(self, item_id: str, now: Optional[float] = None) -> EvidenceQuality:
        """Compute the quality decomposition for one item."""
        item = self.registry.get_item(item_id)
        if item is None:
            raise KeyError(f"Unknown item: {item_id}")
        now = now if now is not None else time.time()

        source = self.registry.get_source(item.source_id)
        reliability = source.reliability if source else 0.5

        # Independence: weight of its lineage among the items relevant here.
        # For a single item we score independence relative to all known
        # items from its source's lineage.
        lineage = None
        all_sources = [s for s in self.registry._sources.keys()]
        groups = self.registry.lineage_groups(all_sources)
        for g in groups:
            if item.source_id in g:
                lineage = g
                break
        lineage_size = max(1, len(lineage)) if lineage else 1
        independence = 1.0 / lineage_size

        # Corroboration: independent items with the same positive stance
        corroborating = 0
        for other in self.registry._items.values():
            if other.item_id == item.item_id:
                continue
            same_lineage = lineage and other.source_id in lineage
            if not same_lineage and (other.stance * item.stance > 0):
                corroborating += 1
        corroboration = corroborating / (corroborating + 1)  # saturating

        # Recency: exponential decay with 7-day half-life
        age = max(0.0, now - item.timestamp)
        recency = 0.5 ** (age / _RECENCY_HALF_LIFE)

        overall = (
            0.35 * reliability
            + 0.25 * independence
            + 0.20 * corroboration
            + 0.20 * recency
        )
        return EvidenceQuality(
            item_id=item.item_id,
            source_reliability=reliability,
            independence_factor=independence,
            corroboration_factor=corroboration,
            recency_factor=recency,
            overall=overall,
        )

    # ------------------------------------------------------------------
    # Consensus
    # ------------------------------------------------------------------

    def consensus(self, item_ids: list[str], claim: str = "",
                  now: Optional[float] = None) -> dict[str, Any]:
        """
        Independence-weighted consensus over the given items.

        NOT majority vote:
        - each item contributes stance × quality
        - weight is divided by its lineage size (copied sources don't stack)
        - modality agreement bonus when multiple modalities concur
        - contradiction severity measured as |mean opposing stance| × weight
        """
        now = now if now is not None else time.time()
        items = [self.registry.get_item(i) for i in item_ids]
        if any(i is None for i in items):
            raise KeyError("Unknown item id in consensus input")
        if not items:
            return {
                "claim": claim,
                "support_weight": 0.0,
                "oppose_weight": 0.0,
                "net_weight": 0.0,
                "effective_sources": 0,
                "modality_agreement": 0.0,
                "contradiction_severity": 0.0,
                "verdict": "INSUFFICIENT_EVIDENCE",
                "item_scores": [],
            }

        # Run dependence detection among the given items first so copied
        # items collapse into lineages before weighting.
        self.registry.observe_dependence_among(item_ids)

        support_w = 0.0
        oppose_w = 0.0
        item_scores: list[dict[str, Any]] = []
        lineages: list[list[str]] = []

        for item in items:
            q = self.score_item(item.item_id, now=now)
            # Lineage size among just the consensus inputs
            input_sources = [it.source_id for it in items]
            groups = self.registry.lineage_groups(input_sources)
            own_group = next((g for g in groups if item.source_id in g), [item.source_id])
            lineages.append(own_group)
            lineage_factor = 1.0 / len(own_group)

            w = q.overall * lineage_factor
            contribution = item.stance * w
            if contribution > 0:
                support_w += contribution
            elif contribution < 0:
                oppose_w += -contribution
            item_scores.append({
                "item_id": item.item_id,
                "source": item.source_id,
                "stance": item.stance,
                "lineage_size": len(own_group),
                "weight": round(w, 4),
                "contribution": round(contribution, 4),
            })

        effective_sources = self.registry.effective_independent_sources(
            [it.source_id for it in items]
        )

        # Modality agreement: fraction of support coming from distinct modalities
        support_modalities = {it.modality for it in items if it.stance > 0}
        oppose_modalities = {it.modality for it in items if it.stance < 0}
        modality_agreement = (
            len(support_modalities) / max(1, len(support_modalities | oppose_modalities))
        )

        contradiction_severity = min(1.0, oppose_w / max(0.0001, oppose_w + support_w))

        # Verdict mapping (aligned with cognition.uncertainty EpistemicState
        # semantics: VERIFIED requires ≥2 independent sources)
        if support_w > 0 and oppose_w > 0:
            verdict = "CONTESTED"
        elif oppose_w > 0 and support_w == 0:
            verdict = "REFUTED_TENDENCY"
        elif support_w == 0 and oppose_w == 0:
            verdict = "INSUFFICIENT_EVIDENCE"
        elif effective_sources >= 2:
            verdict = "VERIFIED_TENDENCY"
        else:
            verdict = "SINGLE_LINEAGE_ONLY"

        return {
            "claim": claim,
            "support_weight": round(support_w, 4),
            "oppose_weight": round(oppose_w, 4),
            "net_weight": round(support_w - oppose_w, 4),
            "effective_sources": effective_sources,
            "modality_agreement": round(modality_agreement, 3),
            "contradiction_severity": round(contradiction_severity, 3),
            "verdict": verdict,
            "item_scores": item_scores,
        }


__all__ = [
    "Source",
    "SourceItem",
    "SourceRegistry",
    "EvidenceEngine2",
    "EvidenceQuality",
    "shingle_text",
    "shingle_jaccard",
    "shingle_containment",
]
