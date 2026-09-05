"""Hardware/software environment detection (Phase 2)."""
from __future__ import annotations

import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _try_import_version(modname: str) -> str:
    try:
        mod = __import__(modname)
        return str(getattr(mod, "__version__", "?"))
    except Exception:
        return "not installed"


def git_info() -> dict:
    info = {"sha": "unknown", "status": "unknown", "branch": "unknown"}
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        if r.returncode == 0:
            info["sha"] = r.stdout.strip()
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=10,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        if r.returncode == 0:
            info["branch"] = r.stdout.strip()
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        info["status"] = "dirty" if r.stdout.strip() else "clean"
    except Exception:
        pass
    return info


def detect() -> dict:
    env: dict = {
        "timestamp": datetime.datetime.now().isoformat(),
        "os": platform.platform(),
        "os_system": platform.system(),
        "os_release": platform.release(),
        "node": platform.node(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "python_impl": platform.python_implementation(),
    }
    try:
        import psutil
        env["cpu_physical_cores"] = psutil.cpu_count(logical=False)
        env["cpu_logical_cores"] = psutil.cpu_count(logical=True)
        vm = psutil.virtual_memory()
        env["ram_total_gb"] = round(vm.total / 1e9, 2)
        env["ram_available_gb"] = round(vm.available / 1e9, 2)
        env["ram_percent_used"] = vm.percent
        try:
            f = psutil.cpu_freq()
            env["cpu_freq_mhz"] = f.current if f else None
        except Exception:
            env["cpu_freq_mhz"] = None
    except Exception as e:
        env["cpu_physical_cores"] = os.cpu_count()
        env["cpu_logical_cores"] = os.cpu_count()
        env["ram_total_gb"] = "unknown"
        env["ram_available_gb"] = "unknown"
        env["psutil_error"] = str(e)

    # GPU
    gpu_lines = []
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=8)
        if r.returncode == 0:
            gpu_lines = [l for l in r.stdout.strip().splitlines() if l]
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=8,
        )
        if r.returncode == 0 and r.stdout.strip():
            gpu_lines += [l for l in r.stdout.strip().splitlines() if l]
    except Exception:
        pass
    env["gpu"] = gpu_lines if gpu_lines else ["none detected"]

    env["versions"] = {
        "torch": _try_import_version("torch"),
        "tensorflow": _try_import_version("tensorflow"),
        "transformers": _try_import_version("transformers"),
        "numpy": _try_import_version("numpy"),
        "sklearn": _try_import_version("sklearn"),
        "psutil": _try_import_version("psutil"),
        "pandas": _try_import_version("pandas"),
        "pytest": _try_import_version("pytest"),
    }
    env["git"] = git_info()
    try:
        env["cpu_brand"] = subprocess.run(
            ["python", "-c", "import platform; print(platform.processor())"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
    except Exception:
        env["cpu_brand"] = platform.processor()

    # Compiler/toolchain presence
    env["compilers"] = {
        "gcc": shutil.which("gcc"),
        "clang": shutil.which("clang"),
        "cc": shutil.which("cc"),
        "cmake": shutil.which("cmake"),
        "make": shutil.which("make"),
    }
    return env


if __name__ == "__main__":
    print(json.dumps(detect(), indent=2))
