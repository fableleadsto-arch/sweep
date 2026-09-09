"""Phase 2 â€” Traceable Evidence Network.

Entity -> Relationship -> Claim -> Evidence -> Source/Provenance.

The graph is persisted through `CognitiveStore`; a query starting from an
entity reconstructs relevant claims plus supporting and contradicting
evidence, with provenance intact.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .schema import Claim, Evidence, EvidentialRelation
from .store import CognitiveStore


@dataclass
class Relationship:
    """A typed edge between two entities."""

    subject_id: str
    predicate: str
    object_id: str
    source: str = ""
    provenance: str = ""
    id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            from .ids import content_id

            self.id = content_id("rel", self.subject_id, self.predicate, self.object_id, self.source)


@dataclass
class Relation:
    """A claim's relation to a piece of evidence (supports/refutes/neutral)."""

    claim_id: str
    evidence_id: str
    relation: EvidentialRelation
    confidence: float = 0.5
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["relation"] = self.relation.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Relation":
        names = {f.name for f in dataclasses.fields(cls)}
        vals = {k: v for k, v in d.items() if k in names}
        vals["relation"] = EvidentialRelation(d["relation"])
        return cls(**vals)


class EvidenceGraph:
    """Persistent evidence network over entities, claims, evidence, sources."""

    def __init__(self, store: CognitiveStore | None = None, store_path=None, in_memory: bool = False):
        self.store = store or CognitiveStore(path=store_path, in_memory=in_memory)
        self._relations: Dict[str, List[str]] = {}  # claim_id -> evidence_ids
        self._relations_meta: Dict[str, Dict[str, str]] = {}  # "claim:evid" -> relation+confidence
        self._relationships: List[Relationship] = []
        self._load_compound()

    def _load_compound(self) -> None:
        for blob in self.store._compound():
            kind = blob.get("_ctype")
            if kind == "relation":
                rel = Relation.from_dict(blob["payload"])
                self.link(rel.claim_id, rel.evidence_id, rel.relation, rel.confidence, rel.reason, persist=False)
            elif kind == "relationship":
                self._relationships.append(Relationship(**blob["payload"]))
        self.store._index_rel_claim_evidence()

    # ------------------------------------------------------------------ API
    def add_entity(self, entity: Any) -> Any:
        return self.store.save(entity)

    def upsert_entity(self, name: str, kind: str = "unknown", **attrs: Any) -> Any:
        from .schema import Entity

        existing = self.store.query("Entity", name=name)
        if existing:
            ent = existing[0]
            for k, v in attrs.items():
                setattr(ent, k, v)
            return self.store.save(ent)
        ent = Entity(name=name, kind=kind, **attrs)
        return self.store.save(ent)

    # ---- claims & evidence ------------------------------------------------
    def add_claim(self, claim: Claim) -> Claim:
        return self.store.save(claim)

    def add_evidence(self, evidence: Evidence) -> Evidence:
        if not evidence.content_hash:
            evidence.content_hash = hashlib.sha256(evidence.content.encode("utf-8")).hexdigest()
        return self.store.save(evidence)

    def link(self, claim_id: str, evidence_id: str, relation: EvidentialRelation,
             confidence: float = 0.5, reason: str = "", persist: bool = True) -> None:
        key = f"{claim_id}:{evidence_id}"
        self._relations.setdefault(claim_id, [])
        if evidence_id not in self._relations[claim_id]:
            self._relations[claim_id].append(evidence_id)
        self._relations_meta[key] = {
            "relation": relation.value,
            "confidence": confidence,
            "reason": reason,
        }
        if persist:
            self.store._compound({"_ctype": "relation", "payload": Relation(
                claim_id, evidence_id, relation, confidence, reason).to_dict()})

    def add_relationship(self, rel: Relationship) -> None:
        self._relationships.append(rel)
        self.store._compound({"_ctype": "relationship", "payload": dataclasses.asdict(rel)})

    # ---- queries ----------------------------------------------------------
    def relations_for(self, claim_id: str) -> List[Relation]:
        out = []
        for eid in self._relations.get(claim_id, []):
            meta = self._relations_meta.get(f"{claim_id}:{eid}", {})
            out.append(Relation(
                claim_id, eid,
                EvidentialRelation(meta.get("relation", "neutral")),
                float(meta.get("confidence", 0.5)),
                meta.get("reason", ""),
            ))
        return out

    def supporting_evidence(self, claim_id: str) -> List[Evidence]:
        return [self.store.get(r.evidence_id) for r in self.relations_for(claim_id)
                if r.relation == EvidentialRelation.SUPPORTS and self.store.get(r.evidence_id)]

    def contradicting_evidence(self, claim_id: str) -> List[Evidence]:
        return [self.store.get(r.evidence_id) for r in self.relations_for(claim_id)
                if r.relation == EvidentialRelation.REFUTES and self.store.get(r.evidence_id)]

    def evidence_with_provenance(self, claim_id: str) -> List[dict]:
        """The milestone chain: Claim -> Evidence -> Source/Provenance."""
        out = []
        for eid in self._relations.get(claim_id, []):
            ev = self.store.get(eid)
            if ev is None:
                continue
            meta = self._relations_meta.get(f"{claim_id}:{eid}", {})
            out.append({
                "evidence": ev,
                "relation": meta.get("relation", "neutral"),
                "relation_confidence": float(meta.get("confidence", 0.5)),
                "source": ev.source,
                "retrieval_time": ev.retrieval_time,
                "content_hash": ev.content_hash,
                "provenance": ev.metadata.get("provenance", ""),
            })
        return out

    def reconstruct_from_entity(self, entity_id: str) -> Dict[str, Any]:
        """From an entity, reconstruct its claims and supporting/contradicting evidence."""
        claims = self.store.query("Claim", **{}) if False else self._claims_for_entity(entity_id)
        result = {"entity_id": entity_id, "claims": []}
        for clm in claims:
            sup = [e.to_dict() for e in self.supporting_evidence(clm.id)]
            con = [e.to_dict() for e in self.contradicting_evidence(clm.id)]
            result["claims"].append({"claim": clm.to_dict(), "supporting": sup, "contradicting": con})
        return result

    def _claims_for_entity(self, entity_id: str) -> List[Claim]:
        claims = []
        for blob_id in self.store._index:
            o = self.store.get(blob_id)
            if isinstance(o, Claim) and entity_id in o.entity_ids:
                claims.append(o)
        return claims

    def relationship_chain(self, entity_id: str, max_depth: int = 3) -> List[Relation]:
        """Traverse Entity -> Relationship edges reachable from entity_id."""
        seen: set = set()
        frontier = [entity_id]
        chains: List[dict] = []
        self._traverse(frontier, seen, chains, max_depth)
        rels = []
        for c in chains:
            for r in c["relations"]:
                rels.append(r)
        return rels

    def _traverse(self, frontier: List[str], seen: set, chains: List[dict], depth: int) -> None:
        if depth <= 0 or not frontier:
            return
        nxt: List[str] = []
        for nid in frontier:
            if nid in seen:
                continue
            seen.add(nid)
            rel_chain = []
            for rel in self._relationships:
                if rel.subject_id == nid:
                    rel_chain.append(rel)
                    nxt.append(rel.object_id)
            chains.append({"node": nid, "relations": rel_chain})
        self._traverse(nxt, seen, chains, depth - 1)

    def claims_for_entity(self, entity_id: str) -> List[Claim]:
        return self._claims_for_entity(entity_id)


# convenience: expose Relation accumulation not needed here
def _noop(obj=None):
    return obj
