"""Fast-path primitives with optional compiled acceleration.

When the ``sweep_native`` extension module has been built (see native/),
its implementations are used directly. Otherwise the pure-Python
fallbacks below run and every call stays correct — only slower.
"""

from typing import Callable

from sweep_core.fastpath import fallback as _fallback

try:
    import sweep_native

    NATIVE_AVAILABLE = True
except ImportError:
    sweep_native = None
    NATIVE_AVAILABLE = False


def rms_energy(samples) -> float:
    impl: Callable = sweep_native.rms_energy if NATIVE_AVAILABLE else _fallback.rms_energy
    return impl(samples)


def moving_average(values, window: int) -> list:
    impl: Callable = sweep_native.moving_average if NATIVE_AVAILABLE else _fallback.moving_average
    return impl(values, window)


def frame_diff_score(prev: bytes, curr: bytes) -> float:
    impl: Callable = (
        sweep_native.frame_diff_score if NATIVE_AVAILABLE else _fallback.frame_diff_score
    )
    return impl(prev, curr)


def dot_product(a, b) -> float:
    impl: Callable = sweep_native.dot_product if NATIVE_AVAILABLE else _fallback.dot_product
    return impl(a, b)


def cosine_similarity(a, b) -> float:
    impl: Callable = (
        sweep_native.cosine_similarity if NATIVE_AVAILABLE else _fallback.cosine_similarity
    )
    return impl(a, b)
