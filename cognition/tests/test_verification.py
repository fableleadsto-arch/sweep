"""Phase 12 — Independent Verification tests."""

import pytest

from cognition.verification import (VerificationEngine, VerificationResult,
                                    VerificationStatus, contested, finalize,
                                    verified)


def test_verified_by_two_independent_checks():
    eng = VerificationEngine()
    eng.register_check("neural", lambda cid: cid == "c-neuro")
    eng.register_check("logic", lambda cid: cid == "c-neuro")
    eng.register_check("evidence", lambda cid: cid == "c-neuro")
    r = eng.run("c-neuro")
    assert r.status is VerificationStatus.VERIFIED
    assert verified(r)


def test_single_check_is_unresolved():
    eng = VerificationEngine()
    eng.register_check("logic", lambda cid: True)
    r = eng.run("anything")
    assert r.status is VerificationStatus.UNRESOLVED  # one path is not enough


def test_disagreement_is_contested():
    eng = VerificationEngine()
    eng.register_check("logic", lambda cid: True)
    eng.register_check("evidence", lambda cid: False)  # evidence disagrees
    r = eng.run("c-argue")
    assert r.status is VerificationStatus.CONTESTED
    assert contested(r)


def test_none_available_is_unresolved():
    eng = VerificationEngine()
    r = eng.run("c-solo")
    assert r.status is VerificationStatus.UNRESOLVED


def test_unhappy_path_marked_as_unrun():
    eng = VerificationEngine()
    eng.register_check("neural", lambda cid: True)
    def broken(cid):
        raise RuntimeError("engine unavailable")
    eng.register_check("evidence", broken)
    r = eng.run("c-broke")
    # only the healthy check ran -> unresolved
    assert r.status is VerificationStatus.UNRESOLVED
    assert "evidence" in r.unrun


def test_finalize_pure():
    good = VerificationResult("a", passed=["neural", "logic"])
    assert finalize(good) is VerificationStatus.VERIFIED

    dispute = VerificationResult("b", passed=["neural"], failed=["evidence"])
    assert finalize(dispute) is VerificationStatus.CONTESTED

    alone = VerificationResult("c", passed=["neural"])
    assert finalize(alone) is VerificationStatus.UNRESOLVED

    nothing = VerificationResult("d")
    assert finalize(nothing) is VerificationStatus.UNRESOLVED


def test_result_to_dict():
    eng = VerificationEngine()
    eng.register_check("logic", lambda cid: True)
    eng.register_check("evidence", lambda cid: True)
    r = eng.run("c-dict")
    d = r.to_dict()
    assert d["status"] == "VERIFIED"
    assert set(d["passed"]) == {"logic", "evidence"}
    assert d["claim_id"] == "c-dict"