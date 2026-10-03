"""Small native OS integrations; no browser or HTTP server is involved."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def data_directory() -> Path:
    override = os.environ.get("SWEEP_DESKTOP_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Sweep"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Sweep"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "sweep"


def launch_command(*arguments: str) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, *arguments]
    executable = Path(sys.executable)
    if sys.platform == "win32" and executable.with_name("pythonw.exe").exists():
        executable = executable.with_name("pythonw.exe")
    return [str(executable), "-m", "sweep.desktop", *arguments]


def set_startup(enabled: bool) -> None:
    if sys.platform != "win32":
        raise RuntimeError("Automatic startup is currently supported on Windows.")
    import winreg
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run") as key:
        if enabled:
            winreg.SetValueEx(key, "Sweep", 0, winreg.REG_SZ,
                             subprocess.list2cmdline(launch_command("--background")))
        else:
            try:
                winreg.DeleteValue(key, "Sweep")
            except FileNotFoundError:
                pass


def startup_enabled() -> bool:
    if sys.platform != "win32":
        return False
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run") as key:
            return bool(winreg.QueryValueEx(key, "Sweep")[0])
    except FileNotFoundError:
        return False
