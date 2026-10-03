from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
import time

import pytest

from services.model_loading import serialized_model_load
from services.intelligence.model_manager.registry import ModelRegistry, ModelStatus


def test_model_initialization_is_serialized_and_reentrant():
    start = Barrier(4)
    counter_lock = Lock()
    active = 0
    peak = 0

    @serialized_model_load
    def nested():
        return 42

    @serialized_model_load
    def initialize():
        nonlocal active, peak
        with counter_lock:
            active += 1
            peak = max(active, peak)
        try:
            time.sleep(0.02)
            return nested()
        finally:
            with counter_lock:
                active -= 1

    def worker():
        start.wait(timeout=5)
        return initialize()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker) for _ in range(4)]
        assert [future.result(timeout=5) for future in futures] == [42] * 4
    assert peak == 1


def test_failed_constructor_releases_lock():
    @serialized_model_load
    def broken():
        raise ValueError("invalid weights")

    @serialized_model_load
    def good():
        return "loaded"

    with pytest.raises(ValueError):
        broken()
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(good).result(timeout=5) == "loaded"


def test_registries_do_not_share_mutable_status(tmp_path):
    first = ModelRegistry(str(tmp_path / "first"))
    second = ModelRegistry(str(tmp_path / "second"))
    original = second.get("bert-base-uncased").status
    first.get("bert-base-uncased").status = ModelStatus.ERROR
    assert second.get("bert-base-uncased").status == original
