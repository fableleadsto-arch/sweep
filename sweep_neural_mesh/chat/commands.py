"""English -> shell command parsing and guarded execution.

The chat base (LoRA'd Qwen) emits either:
  - a direct answer (plain text), or
  - a shell action line:  RUN: <command>

This module validates RUN lines, executes them, and returns captured output
for the engine to summarize. Commands are length-capped and block a small
set of destructive patterns (rm -rf /, fork bombs, dd to devices, etc.) —
the user chose full-shell convenience, this only guards against accidents.
"""
from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass, field

# Destructive patterns that are always refused (accident guard, not a sandbox)
_FORBIDDEN = [
    r"rm\s+-rf\s+/(?:\s|$)",           # rm -rf /
    r":\(\)\s*\{\s*:\|:&\s*\};\s*:",   # fork bomb
    r"mkfs(\.|\s)",                     # format filesystem
    r"dd\s+.*of=/dev/",                 # dd to device
    r">\s*/dev/sd[a-z]",                # overwrite disk device
    r"shutdown|reboot|halt|poweroff",   # host power control
    r"\bformat\b\s+[cC]:",              # windows format
    r"reg\s+delete\s+HK",               # registry deletion
]

MAX_OUTPUT_CHARS = 8000
MAX_COMMAND_SECONDS = 30


@dataclass
class CommandResult:
    ok: bool
    command: str = ""
    output: str = ""
    error: str = ""
    refused_reason: str = ""
    meta: dict = field(default_factory=dict)


def extract_run_line(text: str) -> str | None:
    """Return the command from a 'RUN: ...' line, if present."""
    m = re.search(r"^\s*RUN:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    return m.group(1).strip() if m else None


def is_forbidden(cmd: str) -> str | None:
    low = cmd.lower()
    for pat in _FORBIDDEN:
        if re.search(pat, low):
            return pat
    return None


def _strip_code_fence(cmd: str) -> str:
    m = re.search(r"```(?:bash|sh|shell|powershell|bat|cmd)?\s*\n?(.*?)```",
                  cmd, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return cmd.strip()


def run_command(cmd: str, cwd: str | None = None, timeout: int = MAX_COMMAND_SECONDS) -> CommandResult:
    """Execute a shell command and capture combined output."""
    cmd = _strip_code_fence(cmd)
    if not cmd:
        return CommandResult(ok=False, error="empty command")
    if len(cmd) > 2000:
        return CommandResult(ok=False, command=cmd, error="command too long")
    bad = is_forbidden(cmd)
    if bad:
        return CommandResult(ok=False, command=cmd,
                             refused_reason=f"destructive pattern blocked: {bad}")
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=cwd,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        out = out[:MAX_OUTPUT_CHARS]
        return CommandResult(ok=proc.returncode == 0, command=cmd,
                             output=out, error="" if proc.returncode == 0 else f"exit {proc.returncode}")
    except subprocess.TimeoutExpired:
        return CommandResult(ok=False, command=cmd,
                             error=f"timed out after {timeout}s")
    except Exception as e:
        return CommandResult(ok=False, command=cmd, error=str(e))
