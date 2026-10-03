"""Third-party capability integrations for Sweep.

Each module exposes ``availability()`` describing which optional
dependencies are importable in the current environment. Nothing here
crashes when a dependency is missing — features degrade gracefully.
"""

from __future__ import annotations

from importlib import import_module, util
from typing import Any


def _module_available(name: str) -> bool:
    try:
        return util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _probe(module: str, attr: str | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {"module": module, "available": _module_available(module)}
    if info["available"] and attr:
        try:
            getattr(import_module(module), attr)
            info["entrypoint"] = attr
        except AttributeError:
            info["available"] = False
            info["reason"] = f"{module}.{attr} not found"
    return info


def capabilities() -> dict[str, Any]:
    """Return the live status of every integrated third-party capability."""
    from sweep_core.integrations import audio, bluetooth, resources, scraping, search, vision

    return {
        "scraping": scraping.availability(),
        "audio": audio.availability(),
        "vision": vision.availability(),
        "search": search.availability(),
        "bluetooth": bluetooth.availability(),
        "resources": resources.availability(),
    }
