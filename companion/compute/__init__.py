"""Compute backends package — hardware-aware execution for the brain.

Public entry points:

    BackendManager          discover/probe/enable/disable/install backends
    schedule()              hardware-aware task → backend scheduling
    ModelManager            framework-independent model resolution
    detect()                structured host capability detection
    diagnose()              full diagnostics report (CLI + API)
    FairComputePool         bounded owner-scoped subprocess dispatch

Nothing here imports a heavy framework at import time.
"""

from __future__ import annotations

from .backend_manager import BackendManager
from .base import (
    BackendKind,
    BackendStatus,
    ComputeBackend,
    ComputeJobResult,
    ComputeTask,
    ComputeTaskKind,
    DeviceInfo,
)
from .capability_detector import ComputeEnvironment, detect
from .diagnostics import diagnose
from .model_manager import ModelManager, ModelSpec
from .pool import FairComputePool, QueueFullError
from .scheduler import ScheduleDecision, schedule

__all__ = [
    "BackendKind",
    "BackendManager",
    "BackendStatus",
    "ComputeBackend",
    "ComputeEnvironment",
    "ComputeJobResult",
    "ComputeTask",
    "ComputeTaskKind",
    "DeviceInfo",
    "FairComputePool",
    "ModelManager",
    "ModelSpec",
    "QueueFullError",
    "ScheduleDecision",
    "detect",
    "diagnose",
    "schedule",
]
