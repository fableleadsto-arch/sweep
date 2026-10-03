import math
import random

import pytest

from sweep_core.fastpath import (
    NATIVE_AVAILABLE,
    cosine_similarity,
    dot_product,
    fallback,
    frame_diff_score,
    moving_average,
    rms_energy,
)


class TestFallbackRmsEnergy:
    @pytest.mark.parametrize(
        ("samples", "expected"),
        [
            ([], 0.0),
            ([0.0], 0.0),
            ([3.0, 4.0], math.sqrt(12.5)),
        ],
    )
    def test_known_values(self, samples, expected):
        assert math.isclose(fallback.rms_energy(samples), expected)

    def test_constant_signal(self):
        samples = [0.5] * 1000
        assert math.isclose(fallback.rms_energy(samples), 0.5)


class TestFallbackMovingAverage:
    def test_full_window(self):
        result = fallback.moving_average([1.0, 2.0, 3.0, 4.0], window=2)
        assert result == [1.0, 1.5, 2.5, 3.5]

    def test_warmup_partial_windows(self):
        result = fallback.moving_average([10.0, 20.0, 30.0], window=10)
        assert result == [10.0, 15.0, 20.0]

    def test_empty(self):
        assert fallback.moving_average([], window=4) == []

    @pytest.mark.parametrize("window", [0, -1])
    def test_invalid_window_raises(self, window):
        with pytest.raises(ValueError):
            fallback.moving_average([1.0], window=window)


class TestFallbackFrameDiffScore:
    def test_identical_frames(self):
        frame = bytes(range(256)) * 4
        assert fallback.frame_diff_score(frame, frame) == 0.0

    def test_single_byte_differs(self):
        prev = b"\x00" * 100
        curr = b"\x00" * 99 + b"\xff"
        assert math.isclose(fallback.frame_diff_score(prev, curr), 0.01)

    def test_size_mismatch_counts_as_change(self):
        score = fallback.frame_diff_score(b"\x00\x00\x00\x00", b"\x00")
        assert math.isclose(score, 0.75)

    def test_both_empty(self):
        assert fallback.frame_diff_score(b"", b"") == 0.0

    def test_bounds_on_random_frames(self):
        rng = random.Random(7)
        for _ in range(50):
            a = bytes(rng.randrange(256) for _ in range(64))
            b = bytes(rng.randrange(256) for _ in range(64))
            score = fallback.frame_diff_score(a, b)
            assert 0.0 <= score <= 1.0


class TestVectorOps:
    def test_dot_product_known(self):
        assert math.isclose(dot_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 32.0)

    def test_dot_product_mismatch_raises(self):
        with pytest.raises(ValueError):
            dot_product([1.0], [1.0, 2.0])

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            ([1, 0, 0], [1, 0, 0], 1.0),
            ([1, 0, 0], [0, 1, 0], 0.0),
            ([2, 0], [0, 3], 0.0),
            ([], [], 0.0),
            ([1, 2], [1, 2, 3], 0.0),
            ([0, 0], [1, 1], 0.0),
        ],
    )
    def test_cosine_edge_semantics(self, a, b, expected):
        assert math.isclose(cosine_similarity(a, b), expected)

    def test_cosine_parallel_vectors(self):
        assert math.isclose(cosine_similarity([1.0, 1.0], [2.0, 2.0]), 1.0)

    def test_cosine_opposite_vectors(self):
        assert math.isclose(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)

    def test_dispatcher_matches_fallback(self):
        if NATIVE_AVAILABLE:
            pytest.skip("native module present; fallback equivalence not applicable")
        a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
        assert dot_product(a, b) == fallback.dot_product(a, b)
        assert cosine_similarity(a, b) == fallback.cosine_similarity(a, b)


class TestDispatcher:
    def test_dispatch_matches_fallback_when_native_absent(self):
        if NATIVE_AVAILABLE:
            pytest.skip("native module present; fallback equivalence not applicable")
        frames = (b"abc", b"abd")
        assert rms_energy([1.0, 2.0]) == fallback.rms_energy([1.0, 2.0])
        assert moving_average([1.0, 2.0, 3.0], 2) == fallback.moving_average([1.0, 2.0, 3.0], 2)
        assert frame_diff_score(*frames) == fallback.frame_diff_score(*frames)

    def test_dispatcher_rejects_bad_window(self):
        with pytest.raises(ValueError):
            moving_average([1.0, 2.0], window=0)
