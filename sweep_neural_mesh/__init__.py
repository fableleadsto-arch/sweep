"""Sweep Neural Mesh, with optional orchestration loaded only on demand.

Importing an embedded core must not initialize the full research stack. Existing
public imports remain available when their optional dependencies are installed.
"""
from importlib import import_module

__all__ = ["NeuralMesh", "MeshConstraints", "MeshResult"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".mesh", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
