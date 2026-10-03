"""HTTP layer for keyless engine scraping (Sweep's own stack, no API keys).

Backends, tried in order at first use:
  1. curl-cffi with Chrome TLS impersonation (already a Sweep integration dep)
     — passes the TLS-fingerprint checks most engines apply.
  2. plain httpx with browser headers — fallback when curl-cffi is absent.

GET-page escalation: when a response looks like an anti-bot challenge
(403/429/202 or captcha markers), callers can retry via scrapling's stealth
fetcher (Sweep's existing anti-detect integration) when installed.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Any, Optional

logger = logging.getLogger("sweep.face_search.http")

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

CHALLENGE_MARKERS = (
    "captcha", "unusual traffic", "are you a robot", "enable javascript",
    "just a moment", "access denied", "verify you are human",
)


class Response:
    """Normalized response surface shared by all backends."""

    def __init__(self, *, status_code: int, text: str, content: bytes, url: str = "",
                 headers: Optional[dict[str, str]] = None):
        self.status_code = status_code
        self.text = text
        self.content = content
        self.url = url
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return json.loads(self.text)

    def looks_like_challenge(self) -> bool:
        if self.status_code in (403, 429, 202):
            return True
        low = (self.text[:4000] or "").lower()
        return any(m in low for m in CHALLENGE_MARKERS)


class HttpLike:
    """Minimal session interface (get/post + cookies) over curl-cffi or httpx.

    Thread notes: each engine-scrape provider builds its own instance inside
    its worker thread; a lock guards the lazy backend init anyway.
    """

    def __init__(self, timeout: float = 40.0):
        self.timeout = timeout
        self.backend = ""
        self._session: Any = None
        self._lock = threading.Lock()

    def _ensure(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            try:
                from curl_cffi.requests import Session  # type: ignore

                self._session = Session(impersonate="chrome", timeout=self.timeout)
                self.backend = "curl"
            except Exception:  # noqa: BLE001 — not installed / import failed
                import httpx

                self._session = httpx.Client(
                    timeout=self.timeout,
                    follow_redirects=True,
                    headers=_BROWSER_HEADERS,
                )
                self.backend = "httpx"

    # ── normalized calls ──────────────────────────────────────────────

    def get(self, url: str, **kw: Any) -> Response:
        self._ensure()
        headers = {**_BROWSER_HEADERS, **(kw.pop("headers", None) or {})}
        if self.backend == "curl":
            kw.pop("follow_redirects", None)  # curl-cffi follows by default
            r = self._session.get(url, headers=headers, **kw)
            return Response(
                status_code=r.status_code, text=r.text, content=r.content,
                url=str(getattr(r, "url", url)), headers={
                    k: v for k, v in getattr(r, "headers", {}).items()
                },
            )
        r = self._session.get(url, headers=headers, follow_redirects=True, **kw)
        return Response(
            status_code=r.status_code, text=r.text, content=r.content,
            url=str(r.url), headers={k: v for k, v in r.headers.items()},
        )

    def post(self, url: str, *, data: Optional[dict] = None,
             files: Optional[dict] = None, **kw: Any) -> Response:
        self._ensure()
        headers = {**_BROWSER_HEADERS, **(kw.pop("headers", None) or {})}
        if self.backend == "curl":
            r = self._session.post(url, data=data, files=files, headers=headers, **kw)
            return Response(
                status_code=r.status_code, text=r.text, content=r.content,
                url=str(getattr(r, "url", url)), headers={
                    k: v for k, v in getattr(r, "headers", {}).items()
                },
            )
        r = self._session.post(url, data=data, files=files, headers=headers,
                               follow_redirects=True, **kw)
        return Response(
            status_code=r.status_code, text=r.text, content=r.content,
            url=str(r.url), headers={k: v for k, v in r.headers.items()},
        )

    def close(self) -> None:
        try:
            if self._session is not None:
                self._session.close()
        except Exception:  # noqa: BLE001
            pass


def fetch_html_anti_detect(url: str) -> Optional[str]:
    """Escalate a challenged GET to scrapling's stealth fetcher (best effort).

    Uses Sweep's existing anti-detect integration (camoufox under the hood).
    Returns HTML or None — never raises.
    """
    try:
        from scrapling.fetchers import StealthyFetcher

        page = StealthyFetcher.fetch(url, headless=True, timeout=60_000)
        html = getattr(page, "html_content", None) or getattr(page, "body", None)
        return str(html) if html else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("anti-detect fetch unavailable: %s", exc)
        return None
