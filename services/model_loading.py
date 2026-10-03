"""Coordinate model construction across Sweep's background loaders.

Transformer loading temporarily changes process-wide torch initialization hooks.
Overlapping constructors can leave weights on the meta device. Only construction
is serialized; inference continues independently after a loader returns.
"""

from functools import wraps
from threading import RLock
from typing import Callable, ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")
_MODEL_LOAD_LOCK = RLock()


def serialized_model_load(function: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(function)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with _MODEL_LOAD_LOCK:
            return function(*args, **kwargs)

    return guarded
