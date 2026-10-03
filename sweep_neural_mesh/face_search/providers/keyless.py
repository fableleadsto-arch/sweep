"""Keyless providers — work with no API keys, no public URL hosting.

Two families:

1. Metadata/keyword engines (DuckDuckGo HTML, Wikipedia Commons search).
   These find pages *about* the image context, not pixel matches, so in
   ``face`` mode they only run when the caller supplies a name hint
   (``name_hint="jane dough"``) — querying a person's name against
   platform-specific filters (``site:linkedin.com`` etc.).

2. A public-file fetcher used by the orchestrator to pull candidate
   thumbnails for local face verification (no search endpoint needed).

This is what makes Sweep's scan work with zero credentials: the fan-out
always has at least one runnable source when a name hint is provided, and
the API-keyed providers light up automatically when keys are configured.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from .base import Provider

_DDG_URL = "https://html.duckduckgo.com/html/"
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SweepFaceSearch/1.0"

# Platform → DDG site filter (kept in sync with social_parser.PLATFORMS)
_SITE_FILTERS: dict[str, str] = {
    "linkedin": "site:linkedin.com/in",
    "instagram": "site:instagram.com",
    "facebook": "site:facebook.com",
    "x": "site:x.com OR site:twitter.com",
    "tiktok": "site:tiktok.com/@" ,
    "youtube": "site:youtube.com/@",
    "pinterest": "site:pinterest.com",
    "github": "site:github.com",
    "reddit": "site:reddit.com/user",
}


def _ddg_results(client: Any, query: str, max_results: int) -> list[dict]:
    """Scrape DuckDuckGo's HTML endpoint. Returns [] on any failure — this is
    a best-effort source, never a hard dependency."""
    try:
        resp = client.post(
            _DDG_URL,
            data={"q": query, "kl": "wt-wt"},
            headers={"User-Agent": _UA},
            timeout=20,
        )
        resp.raise_for_status()
    except Exception:  # noqa: BLE001 — keyless source is best-effort
        return []
    html = resp.text
    out: list[dict] = []
    # result blocks: <a rel="nofollow" class="result__a" href="...">
    for m in re.finditer(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        html,
        flags=re.DOTALL,
    ):
        href, title = m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip()
        # DDG wraps URLs: //duckduckgo.com/l/?uddg=<urlencoded>&rut=...
        if "uddg=" in href:
            from urllib.parse import parse_qs, urlparse

            try:
                q = parse_qs(urlparse(href.replace("&amp;", "&")).query)
                href = (q.get("uddg") or [""])[0]
            except Exception:  # noqa: BLE001
                pass
        if not href.startswith("http"):
            continue
        out.append({"link": href, "title": title})
        if len(out) >= max_results:
            break
    return out


class NameHintSearchProvider(Provider):
    """DuckDuckGo text search across platform site-filters.

    Only meaningful in face mode when a name hint is given; runs keyless.
    """

    name = "name_search"
    needs_face = False  # searches by name, not pixels

    def __init__(self, name_hint: str = "", platforms: Optional[list[str]] = None):
        self.name_hint = (name_hint or "").strip()
        self.platforms = platforms or list(_SITE_FILTERS)

    def is_enabled(self) -> bool:
        return bool(self.name_hint)

    def note(self) -> Optional[str]:
        if not self.name_hint:
            return "Runs only when a name hint is provided (searches by name, not pixels)."
        return None

    def search(
        self,
        client: Any,
        *,
        image_url: Optional[str],
        image_bytes: Optional[bytes],
        image_filename: Optional[str],
        max_results: int,
    ) -> Any:
        from ..models import ProviderResult

        if not self.is_enabled():
            return ProviderResult(
                provider=self.name,
                error="no name hint provided (keyless name search disabled)",
            )
        per_platform = max(2, max_results // max(1, len(self.platforms)))
        matches: list[dict] = []
        quoted = f'"{self.name_hint}"'
        for platform, site in _SITE_FILTERS.items():
            for r in _ddg_results(client, f"{site} {quoted}", per_platform):
                r["_platform"] = platform
                matches.append(r)
        return self._result(self.name, matches, max_results)


class CommonsSearchProvider(Provider):
    """Wikipedia/Commons file search — free, keyless, good for public-figure
    photos. Finds pages using a similarly-named image file."""

    name = "commons_search"
    accepts_url = False

    def __init__(self, name_hint: str = ""):
        self.name_hint = (name_hint or "").strip()

    def is_enabled(self) -> bool:
        return bool(self.name_hint)

    def note(self) -> Optional[str]:
        if not self.name_hint:
            return "Runs only when a name hint is provided."
        return "Finds Wikipedia/Commons usage of similarly-named files (public figures only)."

    def search(
        self,
        client: Any,
        *,
        image_url: Optional[str],
        image_bytes: Optional[bytes],
        image_filename: Optional[str],
        max_results: int,
    ) -> Any:
        from ..models import ProviderResult

        if not self.is_enabled():
            return ProviderResult(provider=self.name, error="no name hint provided")
        try:
            data = self._get_json(
                client,
                "https://commons.wikimedia.org/w/api.php",
                {
                    "action": "query",
                    "list": "search",
                    "srsearch": self.name_hint,
                    "srnamespace": "6",  # File:
                    "srlimit": str(max_results),
                    "format": "json",
                },
            )
        except Exception as exc:  # noqa: BLE001
            return ProviderResult(provider=self.name, error=str(exc)[:300])
        items = []
        for hit in (data.get("query") or {}).get("search") or []:
            title = hit.get("title") or ""
            items.append(
                {
                    "link": f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
                    "title": title,
                }
            )
        return self._result(self.name, items, max_results)


KEYLESS_PROVIDERS: list[type[Provider]] = [NameHintSearchProvider, CommonsSearchProvider]


def fetch_image_bytes(client: Any, url: str, *, max_bytes: int = 8 * 1024 * 1024) -> Optional[bytes]:
    """Download an image (thumbnail fetch for face verification). Returns
    None on any failure — callers treat thumbnails as optional."""
    try:
        resp = client.get(url, headers={"User-Agent": _UA}, timeout=20, follow_redirects=True)
        resp.raise_for_status()
        data = resp.content
        if not data or len(data) > max_bytes:
            return None
        return data
    except Exception:  # noqa: BLE001
        return None
