"""Security guard — SSRF protection, URL validation, injection scanning.

Treats every webpage as untrusted input:
  - SSRF guard: never fetch private/internal hosts
  - Parameter validation for controlled tools
  - Prompt-injection scan: web content must never override system policies
"""

from __future__ import annotations

import re
import ipaddress
import socket
from urllib.parse import urlparse

from .types import InjectionAssessment

# ── SSRF Guard ────────────────────────────────────────────────────────

BLOCKED_SCHEMES = frozenset({"file:", "ftp:", "gopher:", "data:", "javascript:", "vbscript:"})


def is_private_host(hostname: str) -> bool:
    """Check if a hostname is private/internal and should never be fetched."""
    h = hostname.lower().rstrip(".")
    if h == "localhost" or h.endswith((".localhost", ".internal", ".local", ".localdomain")):
        return True
    try:
        address = ipaddress.ip_address(h)
    except ValueError:
        # Reject legacy decimal/octal/hex IPv4 spellings accepted by OS resolvers.
        if re.fullmatch(r"[0-9.]+", h) or re.match(r"^0x[0-9a-f.]+$", h):
            return True
    else:
        return not address.is_global or address.is_multicast
    if h in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        return True
    if h.startswith("192.168."):
        return True
    if h.startswith("10."):
        return True
    if h.startswith("172."):
        parts = h.split(".")
        if len(parts) > 1:
            try:
                second = int(parts[1])
                if 16 <= second <= 31:
                    return True
            except ValueError:
                pass
    if h.endswith((".internal", ".local", ".localdomain")):
        return True
    return False


def validate_safe_url(raw: str) -> tuple[bool, str, str]:
    """Validate a URL for fetching. Returns (ok, url, reason)."""
    try:
        parsed = urlparse(raw)
        if any(ord(c) < 33 for c in raw) or "\\" in raw or "%" in (parsed.hostname or ""):
            return False, "", "Invalid characters in URL"
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            return False, "", "Invalid port"
    except Exception:
        return False, "", "Not a valid URL"

    if not parsed.scheme or not parsed.hostname:
        return False, "", "Not a valid URL"

    if parsed.scheme.lower() in BLOCKED_SCHEMES:
        return False, "", f"Scheme not allowed: {parsed.scheme}"

    if parsed.scheme not in ("http", "https"):
        return False, "", "Only http(s) URLs can be fetched"

    if is_private_host(parsed.hostname):
        return False, "", "Private or internal hosts are not reachable"

    if parsed.username or parsed.password:
        return False, "", "URLs with embedded credentials are not allowed"

    return True, raw, ""


def resolve_public_addresses(raw: str) -> list[str]:
    """Resolve once; callers connect to a returned IP to prevent DNS rebinding."""
    parsed = urlparse(assert_safe_url(raw))
    try:
        records = socket.getaddrinfo(
            parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise ValueError("Could not resolve public host") from exc
    addresses = list(dict.fromkeys(record[4][0] for record in records))
    if not addresses or any(is_private_host(address) for address in addresses):
        raise ValueError("Host resolves to a non-public address")
    return addresses


def assert_safe_url(raw: str, fallback_label: str = "url") -> str:
    """Validate a URL and raise if unsafe. Returns the validated URL string."""
    value = raw.strip()
    if not value:
        raise ValueError(f"{fallback_label} is required")
    ok, url, reason = validate_safe_url(value)
    if not ok:
        raise ValueError(f"Unsafe {fallback_label}: {reason}")
    return url


# ── Injection Detection ───────────────────────────────────────────────

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?prior", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an|the)", re.IGNORECASE),
    re.compile(r"system\s*prompt\s*(override|injection|bypass)", re.IGNORECASE),
    re.compile(r"<\|im_start\|>system", re.IGNORECASE),
    re.compile(r"<\|system\|>", re.IGNORECASE),
    re.compile(r"IMPORTANT:\s*you\s+must", re.IGNORECASE),
    re.compile(r"new\s+instructions?:", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all)", re.IGNORECASE),
    re.compile(r"act\s+as\s+if\s+you\s+(have|are)", re.IGNORECASE),
]


def assess_injection(text: str) -> InjectionAssessment:
    """Scan page text for prompt-injection-shaped patterns.

    This is a heuristic warning, not a sandbox. The extraction pipeline
    always quotes content inside a fixed system prompt that explicitly
    tags web content as untrusted data.
    """
    signals: list[str] = []
    sample = text[:5000].lower()
    for pattern in INJECTION_PATTERNS:
        if pattern.search(sample):
            signals.append(pattern.pattern)
    return InjectionAssessment(suspect=len(signals) > 0, signals=signals)


# ── Clamping ──────────────────────────────────────────────────────────

def clamp_int(value: object, fallback: int, min_val: int, max_val: int) -> int:
    """Clamp an integer within bounds."""
    try:
        n = int(float(str(value)))
    except (ValueError, TypeError):
        return fallback
    return max(min_val, min(max_val, n))
