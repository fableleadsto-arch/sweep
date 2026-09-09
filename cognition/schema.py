"""Phase 1 â€” Unified Cognitive Data Contract.

Every major cognitive object lives here with a stable schema, stable identity,
and lossless (de)serialization. The complete structure (`Observation ->
Evidence -> Claim -> Hypothesis -> ReasoningEvent`) survives a JSON
round-trip without losing identity or relationships.
"""

from __future__ import annotations

import dataclasses
import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .ids import new_id

VERSION = "1.0.0"


class EvidentialRelation(str, enum.Enum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class HypothesisStatus(str, enum.Enum):
    ACTIVE = "active"
    SUPPORTED = "supported"
    WEAK = "weak"
    DISMISSED = "dismissed"
    ARCHIVED = "archived"


class EventKind(str, enum.Enum):
    OBSERVATION_RECORDED = "observation_recorded"
    EVIDENCE_LINKED = "evidence_linked"
    CLAIM_FORMED = "claim_formed"
    HYPOTHESIS_FORMED = "hypothesis_formed"
    HYPOTHESIS_UPDATED = "hypothesis_updated"
    CONTRADICTION_DETECTED = "contradiction_detected"
    BELIEF_REVISED = "belief_revised"
    VERIFICATION = "verification"
    ROUTING = "routing"
    RULE_APPLIED = "rule_applied"


def _now() -> float:
    return time.time()


def _field_names(cls: type) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


@dataclass
class Observation:
    """A raw percept: something observed outside the reasoner."""

    content: str
    modality: str = "text"
    actor: str = ""
    location: str = ""
    obs_time: Optional[str] = None
    provenance: str = ""
    id: str = field(default_factory=lambda: new_id("obs"))
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Observation":
        return cls(**{k: v for k, v in d.items() if k in _field_names(cls)})


@dataclass
class Entity:
    """A thing the system reasons about."""

    name: str
    kind: str = "unknown"
    attributes: Dict[str, Any] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: new_id("ent"))
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Entity":
        return cls(**{k: v for k, v in d.items() if k in _field_names(cls)})


@dataclass
class Evidence:
    """A piece of evidence bearing on a claim."""

    content: str
    source: str = ""
    retrieval_time: str = ""
    content_hash: str = ""
    entity_ids: list[str] = field(default_factory=list)
    observation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("ev"))
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Evidence":
        return cls(**{k: v for k, v in d.items() if k in _field_names(cls)})


@dataclass
class Claim:
    """A statement asserted about entities."""

    statement: str
    subject: str = ""
    predicate: str = ""
    object: str = ""
    time_context: str = ""
    confidence: float = 0.5
    status: str = "unverified"
    evidence_ids: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: new_id("clm"))
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Claim":
        return cls(**{k: v for k, v in d.items() if k in _field_names(cls)})


@dataclass
class Hypothesis:
    """A candidate explanation, held in competition with others."""

    statement: str
    confidence: float = 0.5
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    claim_ids: list[str] = field(default_factory=list)
    notes: str = ""
    id: str = field(default_factory=lambda: new_id("hyp"))
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Hypothesis":
        vals = {k: v for k, v in d.items() if k in _field_names(cls)}
        if "status" in d and isinstance(d["status"], str):
            vals["status"] = HypothesisStatus(d["status"])
        return cls(**vals)


@dataclass
class ReasoningEvent:
    """A recorded step of reasoning that can be replayed for explanation."""

    kind: EventKind
    detail: str = ""
    object_type: str = ""
    object_id: str = ""
    parent_event_id: Optional[str] = None
    evidence_ids: list[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("evt"))
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["kind"] = self.kind.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReasoningEvent":
        vals = {k: v for k, v in d.items() if k in _field_names(cls)}
        if "kind" in d and isinstance(d["kind"], str):
            vals["kind"] = EventKind(d["kind"])
        return cls(**vals)


_CONTRACT = {
    "Observation": Observation,
    "Entity": Entity,
    "Evidence": Evidence,
    "Claim": Claim,
    "Hypothesis": Hypothesis,
    "ReasoningEvent": ReasoningEvent,
}


def object_to_dict(obj: Any) -> Dict[str, Any]:
    """Serialize any contract object with its type tag."""
    d = obj.to_dict()
    d["_type"] = obj.__class__.__name__
    d["_schema_version"] = VERSION
    return d


def object_from_dict(d: Dict[str, Any]) -> Any:
    """Deserialize any contract object, preserving identity and relationships."""
    tname = d.pop("_type", None)
    d.pop("_schema_version", None)
    if tname not in _CONTRACT:
        raise ValueError(f"Unknown contract type: {tname}")
    return _CONTRACT[tname].from_dict(d)


def roundtrip(obj: Any) -> Any:
    """Serialize then deserialize; the result preserves identity + relationships."""
    restored = object_from_dict(object_to_dict(obj))
    assert restored.id == obj.id, "identity lost in round-trip"
    return restored
