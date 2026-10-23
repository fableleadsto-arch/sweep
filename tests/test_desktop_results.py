"""Saved evidence remains bounded, durable and recoverable across consumers."""
import json
from pathlib import Path

import pytest

from sweep.desktop import results
from sweep.desktop.results import MAX_RESULT_BYTES, load_result, save_result
from sweep.desktop.tasks import TaskStore


def running(store):
    task = store.create("conversation", "Remember the shipment quantity")
    return store.update(task["id"], state="running", approved=True)


def test_completed_result_round_trip_and_retry_never_overwrites_evidence(tmp_path):
    store = TaskStore(tmp_path)
    task = running(store)
    result = {"message": "Quantity: 12 — प्रमाण", "rows": [[12]], "columns": ["Quantity"]}
    completed = store.complete(task["id"], result)
    assert completed["state"] == "completed" and completed["approved"] is False
    reopened = TaskStore(tmp_path)
    assert reopened.result(task["id"]) == result
    result["rows"][0][0] = 10
    assert reopened.result(task["id"])["rows"] == [[12]]
    with pytest.raises(ValueError, match="running"):
        store.complete(task["id"], result)
    with pytest.raises(ValueError, match="already"):
        save_result(store.directory(task["id"]) / "result.json", result)


@pytest.mark.parametrize("result", [[], {"message": []}, {"location": None}, {"analysis": []},
    {"evidence": [False]}, {"sources": [{"url": 123}]}, {"columns": [1]}, {"rows": [1]},
    {"value": float("nan")}, {"value": float("inf")}, {1: "coerced"}, {"value": object()},
    {"context_sources": [{}]}, {"bytes": -1}, {"location": {"status": "embedded_gps"}}])
def test_invalid_results_cannot_complete_or_leave_temporary_files(tmp_path, result):
    store = TaskStore(tmp_path)
    task = running(store)
    with pytest.raises(ValueError):
        store.complete(task["id"], result)
    assert store.get(task["id"])["state"] == "running"
    assert [p.name for p in store.directory(task["id"]).iterdir()] == ["task.json"]


@pytest.mark.parametrize("raw", [b'{"message":"first","message":"second"}',
    b'{"value":NaN}', b'{"value":1e999}', b'{"rows":[{}]}',
    b'{"value":' + b'[' * 100 + b'0' + b']' * 100 + b'}'])
def test_malformed_results_fail_consistently_on_read(tmp_path, raw):
    path = tmp_path / "result.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="damaged"):
        load_result(path)


def test_utf8_byte_limit_is_exact_on_write_and_read(tmp_path):
    path = tmp_path / "result.json"
    overhead = len(b'{"text":""}')
    text = "é" * ((MAX_RESULT_BYTES - overhead) // 2)
    result = {"text": text + "x" * ((MAX_RESULT_BYTES - overhead) % 2)}
    save_result(path, result)
    assert path.stat().st_size == MAX_RESULT_BYTES
    assert load_result(path) == result
    other = tmp_path / "oversized.json"
    with pytest.raises(ValueError, match="2 MB"):
        save_result(other, {"text": result["text"] + "é"})
    assert not other.exists() and not list(tmp_path.glob("*.tmp"))


def test_cycles_and_excessive_structure_fail_before_serialization(tmp_path):
    result = {}
    result["cycle"] = result
    with pytest.raises(ValueError, match="nested"):
        save_result(tmp_path / "result.json", result)
    with pytest.raises(ValueError, match="many values"):
        save_result(tmp_path / "result.json", {"values": [0] * results.MAX_RESULT_NODES})
    assert not list(tmp_path.iterdir())


def test_result_write_failure_does_not_mark_completed(tmp_path, monkeypatch):
    store = TaskStore(tmp_path)
    task = running(store)

    def disk_full(*args):
        raise OSError("disk full")

    monkeypatch.setattr(results.os, "fsync", disk_full)
    with pytest.raises(OSError):
        store.complete(task["id"], {"message": "not durable"})
    assert store.get(task["id"])["state"] == "running"
    assert not (store.directory(task["id"]) / "result.json").exists()
    assert not list(store.directory(task["id"]).glob("*.tmp"))


def test_crash_between_result_and_completion_recovers_as_interrupted(tmp_path, monkeypatch):
    store = TaskStore(tmp_path)
    task = running(store)

    def crash(*args, **kwargs):
        raise OSError("lost metadata write")

    monkeypatch.setattr(store, "update", crash)
    with pytest.raises(OSError):
        store.complete(task["id"], {"message": "observation saved"})
    reopened = TaskStore(tmp_path)
    reopened.recover_interrupted()
    assert reopened.get(task["id"])["state"] == "interrupted"
    with pytest.raises(ValueError, match="no completed"):
        reopened.result(task["id"])
    assert load_result(reopened.directory(task["id"]) / "result.json")["message"] == "observation saved"


def test_same_size_replacement_during_read_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"message": "before"}), encoding="utf-8")
    original = Path.lstat
    reads = 0

    def replace_on_second_stat(candidate):
        nonlocal reads
        if candidate == path:
            reads += 1
            if reads == 2:
                replacement = tmp_path / "replacement.json"
                replacement.write_text(json.dumps({"message": "after!"}), encoding="utf-8")
                replacement.replace(path)
        return original(candidate)

    monkeypatch.setattr(Path, "lstat", replace_on_second_stat)
    with pytest.raises((ValueError, PermissionError)):
        load_result(path)
