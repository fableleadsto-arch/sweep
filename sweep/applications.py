"""Bounded Windows application discovery and native launch receipts.

Only installed entries or a single resolved executable are launched. User text
never becomes a command line, PowerShell script or an AppsFolder identifier.
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import subprocess
import time
from ctypes import wintypes
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


class AmbiguousApplication(ValueError):
    """Several installed applications match the requested name."""


def matching_entry(name: str, entries: tuple[tuple[str, str], ...]) -> str | None:
    """Prefer exact names; otherwise require one whole-word match, not a prefix."""
    def words(value: str) -> tuple[str, ...]:
        return tuple(re.findall(r"\w+", value.casefold()))

    query = words(name)
    if not query:
        return None
    exact = [(label, target) for label, target in entries if words(label) == query]
    matches = exact or [(label, target) for label, target in entries
                        if set(query) <= set(words(label))]
    # Only collapse identical targets. Two different shortcuts with the same
    # display name can launch different programs, so do not silently pick one.
    distinct = {target.casefold(): (label, target) for label, target in matches}
    if len(distinct) == 1:
        return next(iter(distinct.values()))[1]
    if len(distinct) > 1:
        labels = ", ".join(sorted({label for label, _ in distinct.values()})[:6])
        raise AmbiguousApplication(f"Several installed apps match {name!r}: {labels}. Say the full app name or executable path.")
    return None


@lru_cache(maxsize=1)
def windows_start_apps() -> tuple[tuple[str, str], ...]:
    """Include Store apps absent from .lnk files, using a fixed read-only query."""
    if os.name != "nt":
        return ()
    system = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    powershell = system / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    script = ("[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); "
              "Get-StartApps | Select-Object Name,AppID | ConvertTo-Json -Compress")
    try:
        completed = subprocess.run(
            [str(powershell), "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
            shell=False, creationflags=0x08000000, capture_output=True,
            encoding="utf-8", errors="replace", timeout=6,
        )
        if completed.returncode or len(completed.stdout) > 1_000_000:
            return ()
        rows = json.loads(completed.stdout)
        if isinstance(rows, dict):
            rows = [rows]
        if not isinstance(rows, list):
            return ()
        entries = []
        for row in rows[:5000]:
            if not isinstance(row, dict):
                continue
            label, app_id = row.get("Name"), row.get("AppID")
            # AppsFolder is the native Windows registered-app namespace. Never
            # concatenate a user-provided ID or a shell command into this target.
            if (isinstance(label, str) and 0 < len(label) <= 200
                    and isinstance(app_id, str) and 0 < len(app_id) <= 512
                    and not any(ord(char) < 32 or char in '"<>|;' for char in app_id)):
                entries.append((label, "shell:AppsFolder\\" + app_id))
        return tuple(entries)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return ()


@dataclass(frozen=True)
class LaunchReceipt:
    accepted: bool
    window_visible: bool = False
    process_id: int | None = None
    error: str = ""

    def __bool__(self) -> bool:
        return self.accepted


def _visible_window(process_id: int) -> bool:
    """Check a window owned by this process, never infer success from a title."""
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    found = False

    @callback_type
    def visit(window, _):
        nonlocal found
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(window, ctypes.byref(pid))
        if pid.value == process_id and user32.IsWindowVisible(window):
            found = True
            return False
        return True

    user32.EnumWindows(visit, 0)
    return found


def launch_windows(target: str, *, verify_seconds: float = 1.5) -> LaunchReceipt:
    """Use ShellExecuteEx without parameters and observe its process when available.

    Store apps and single-instance apps may hand off to another process. In that
    case the receipt says accepted, not that an unobserved window was opened.
    """
    if os.name != "nt" or not target or any(ord(char) < 32 for char in target):
        return LaunchReceipt(False, error="Invalid native application target.")

    class ShellExecuteInfo(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD), ("fMask", wintypes.ULONG), ("hwnd", wintypes.HWND),
            ("lpVerb", wintypes.LPCWSTR), ("lpFile", wintypes.LPCWSTR),
            ("lpParameters", wintypes.LPCWSTR), ("lpDirectory", wintypes.LPCWSTR),
            ("nShow", ctypes.c_int), ("hInstApp", wintypes.HINSTANCE),
            ("lpIDList", ctypes.c_void_p), ("lpClass", wintypes.LPCWSTR),
            ("hkeyClass", wintypes.HKEY), ("dwHotKey", wintypes.DWORD),
            ("hIcon", wintypes.HANDLE), ("hProcess", wintypes.HANDLE),
        ]

    shell = ctypes.WinDLL("shell32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    shell.ShellExecuteExW.argtypes = [ctypes.POINTER(ShellExecuteInfo)]
    shell.ShellExecuteExW.restype = wintypes.BOOL
    kernel.GetProcessId.argtypes = [wintypes.HANDLE]
    kernel.GetProcessId.restype = wintypes.DWORD
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    info = ShellExecuteInfo()
    info.cbSize = ctypes.sizeof(info)
    # NOCLOSEPROCESS | NOASYNC | FLAG_NO_UI: receive a process handle when the
    # shell provides one and report failures without an extra system error box.
    info.fMask = 0x00000040 | 0x00000100 | 0x00000400
    info.lpVerb, info.lpFile, info.nShow = "open", target, 1
    if not shell.ShellExecuteExW(ctypes.byref(info)):
        return LaunchReceipt(False, error=ctypes.FormatError(ctypes.get_last_error()).strip())
    if not info.hProcess:
        return LaunchReceipt(True)
    try:
        pid = kernel.GetProcessId(info.hProcess)
        deadline = time.monotonic() + max(0, min(verify_seconds, 3))
        while True:
            if pid and _visible_window(pid):
                return LaunchReceipt(True, True, pid)
            exit_code = wintypes.DWORD()
            if kernel.GetExitCodeProcess(info.hProcess, ctypes.byref(exit_code)) and exit_code.value != 259:
                return LaunchReceipt(exit_code.value == 0, process_id=pid or None,
                                     error=f"The application exited with code {exit_code.value}." if exit_code.value else "")
            if time.monotonic() >= deadline:
                return LaunchReceipt(True, process_id=pid or None)
            time.sleep(0.05)
    finally:
        kernel.CloseHandle(info.hProcess)
