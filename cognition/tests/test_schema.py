"""Phase 1 — Unified Cognitive Data Contract tests."""

import pytest

from cognition.schema import (
    Claim,
    Entity,
    Evidence,
    Hypothesis,
    HypothesisStatus,
    Observation,
    ReasoningEvent,
    EvidentialRelation,
    EventKind,
    object_from_dict,
    object_to_dict,
    roundtrip,
)


def _build_chain():
    obs = Observation(content="Agent was seen in the server room at 14:00", location="server room")
    ent = Entity(name="alice", kind="person")
    ev1 = Evidence(content="CCTV shows alice entering at 14:00", source="camera-2", entity_ids=[ent.id])
    ev2 = Evidence(
        content="Access log shows badge 4481 used at 14:01",
        source="access-log",
        entity_ids=[ent.id],
    )
    clm = Claim(
        statement="alice was in the server room at 14:00",
        subject="alice",
        predicate="located-at",
        object="server room",
        evidence_ids=[ev1.id, ev2.id],
        entity_ids=[ent.id],
    )
    hyp = Hypothesis(statement="alice caused the data exfiltration", claim_ids=[clm.id])
    evt = ReasoningEvent(kind=EventKind.HYPOTHESIS_FORMED, object_id=hyp.id, data={"claim_ids": [clm.id]})
    return obs, ent, ev1, ev2, clm, hyp, evt


def test_all_objects_exist():
    obs, ent, ev1, ev2, clm, hyp, evt = _build_chain()
    assert obs.id and ent.id and ev1.id and clm.id and hyp.id and evt.id


def test_roundtrip_preserves_identity_and_relationships():
    obs, ent, ev1, ev2, clm, hyp, evt = _build_chain()
    for o in [obs, ent, ev1, ev2, clm, hyp, evt]:
        restored = roundtrip(o)
        assert restored.id == o.id, f"identity lost for {type(o).__name__}"
    clm_r = roundtrip(clm)
    assert clm_r.evidence_ids == clm.evidence_ids
    assert clm_r.entity_ids == clm.entity_ids
    assert set(clm_r.evidence_ids) == {ev1.id, ev2.id}


def test_serialization_keeps_type_tags():
    evt = ReasoningEvent(kind=EventKind.OBSERVATION_RECORDED, detail="d")
    blob = object_to_dict(evt)
    assert blob["_type"] == "ReasoningEvent"
    assert blob["_schema_version"]
    revived = object_from_dict(dict(blob))
    assert isinstance(revived, ReasoningEvent)
    assert revived.kind == EventKind.OBSERVATION_RECORDED


def test_enum_fields_survive_roundtrip():
    hyp = Hypothesis(statement="x", status=HypothesisStatus.SUPPORTED)
    revived = roundtrip(hyp)
    assert revived.status == HypothesisStatus.SUPPORTED
    assert isinstance(revived.status, HypothesisStatus)


def test_unknown_type_rejected():
    with pytest.raises(ValueError):
        object_from_dict({"_type": "Bogus", "id": "x"})


def test_roundtrip_restores_full_content():
    obs = Observation(content="sensor 7 read 42", modality="sensor", location="lab", obs_time="2026-09-08T10:00:00Z")
    r = roundtrip(obs)
    assert r.content == obs.content
    assert r.location == obs.location
    assert r.obs_time == obs.obs_time
    ev = Evidence(content="e", source="s", entity_ids=["ent-a"])
    r2 = roundtrip(ev)
    assert r2.entity_ids == ["ent-a"]
    ent = Entity(name="x", attributes={"color": "blue"}, aliases=["y"])
    r3 = roundtrip(ent)
    assert r3.attributes == {"color": "blue"}
    assert r3.aliases == ["y"]


def test_observation_to_evidence_to_claim_to_hypothesis_pipeline():
    obs, ent, ev1, ev2, clm, hyp, evt = _build_chain()
    # the milestone flow: Observation -> Evidence -> Claim -> Hypothesis -> ReasoningEvent
    flow = []
    ev_for_obs = [ev1, ev2]
    flow.append(("Observation", obs))
    for ev in ev_for_obs:
        flow.append(("Evidence", ev))
    flow.append(("Claim", clm))
    flow.append(("Hypothesis", hyp))
    flow.append(("ReasoningEvent", evt))

    # every link survives serialization applied at each stage
    restored_ev = [roundtrip(ev) for ev in ev_for_obs]
    restored_clm = roundtrip(clm)
    assert [e.id for e in restored_ev] == restored_clm.evidence_ids
    assert restored_clm.id in roundtrip(hyp).claim_ids