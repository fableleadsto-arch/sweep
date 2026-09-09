"""Phase 6 — Structured Neural-to-Reasoning Integration tests."""

import pytest

from cognition.bridge import NeuralBridge, NeuralOutput, ValidationResult
from cognition.schema import Claim, Evidence, Entity


def _valid_claim(conf=0.9):
    return NeuralOutput(raw="alice was in the server room",
                        structured={"statement": "alice was in the server room", "subject": "alice"},
                        confidence=conf, kind="claim", model_id="bert-01")


def test_valid_claim_accepted_and_transformed():
    b = NeuralBridge(min_confidence=0.7)
    res = b.validate(_valid_claim())
    assert res.accepted
    assert isinstance(res.transformed, Claim)
    assert res.transformed.statement == "alice was in the server room"
    assert res.transformed.subject == "alice"


def test_low_confidence_rejected():
    b = NeuralBridge(min_confidence=0.7)
    res = b.validate(_valid_claim(conf=0.5))
    assert not res.accepted
    assert "confidence" in res.reason.lower()


def test_malformed_missing_field_rejected():
    b = NeuralBridge(min_confidence=0.7)
    out = NeuralOutput(raw="x", structured={}, confidence=0.9, kind="claim", model_id="m")
    res = b.validate(out)
    assert not res.accepted
    assert "missing" in res.reason.lower()


def test_emptyraw_rejected():
    b = NeuralBridge(min_confidence=0.7)
    out = NeuralOutput(raw="   ",
                       structured={"statement": "x"}, confidence=0.9, kind="claim", model_id="m")
    res = b.validate(out)
    assert not res.accepted


def test_non_dict_structured_rejected():
    b = NeuralBridge(min_confidence=0.7)
    out = NeuralOutput(raw="x", structured=[1, 2], confidence=0.9, kind="claim", model_id="m")
    res = b.validate(out)
    assert not res.accepted


def test_unknown_kind_rejected():
    b = NeuralBridge(min_confidence=0.7)
    out = NeuralOutput(raw="x", structured={"statement": "x"}, confidence=0.9, kind="fax", model_id="m")
    res = b.validate(out)
    assert not res.accepted


def test_evidence_and_entity_transform():
    b = NeuralBridge(min_confidence=0.7)
    ev_res = b.validate(NeuralOutput(raw="feed shows it", structured={"content": "feed shows it", "source": "camera-7"},
                                     confidence=0.85, kind="evidence", model_id="m"))
    assert ev_res.accepted and isinstance(ev_res.transformed, Evidence)
    ent_res = b.validate(NeuralOutput(raw="alice", structured={"name": "alice", "kind": "person", "role": "admin"},
                                      confidence=0.9, kind="entity", model_id="m"))
    assert ent_res.accepted and isinstance(ent_res.transformed, Entity)
    assert ent_res.transformed.attributes["role"] == "admin"


def test_malformed_output_never_becomes_trusted_fact():
    """The milestone gate: malformed/low-conf neural output cannot become a trusted fact."""
    b = NeuralBridge(min_confidence=0.8)
    bad_cases = [
        NeuralOutput(raw="x", structured={}, confidence=0.99, kind="claim", model_id="m"),
        NeuralOutput(raw="x", structured={"statement": "y"}, confidence=0.1, kind="claim", model_id="m"),
        NeuralOutput(raw="x", structured={"content": "y", "source": "s"}, confidence=0.6, kind="evidence", model_id="m"),
        NeuralOutput(raw="", structured={"name": "z"}, confidence=0.9, kind="entity", model_id="m"),
    ]
    accepted = 0
    for out in bad_cases:
        r = b.validate(out)
        if r.accepted:
            accepted += 1
    assert accepted == 0
    # logging records routing-of-failure for audit
    assert any(not e["accepted"] for e in b.log)


def test_neural_to_evidence_claim_link():
    b = NeuralBridge(min_confidence=0.7)
    ev = b.validate(NeuralOutput(raw="access log shows badge 4481",
                                 structured={"content": "access log shows badge 4481", "source": "access-log"},
                                 confidence=0.9, kind="evidence", model_id="m")).transformed
    clm = b.validate(NeuralOutput(raw="alice entered",
                                  structured={"statement": "alice entered", "subject": "alice"},
                                  confidence=0.88, kind="claim", model_id="m")).transformed
    assert ev.source == "access-log"
    assert clm.statement == "alice entered"