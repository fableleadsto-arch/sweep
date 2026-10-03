"""Provider contract for the face / reverse-image search engine.

A provider wraps one external search source. The design (taken from
openface-search + selfwatch) keeps every failure as data: a provider that
errors returns ``ProviderResult(error=...)`` and the fan-out continues.

Two engine modes decide which providers run:

  - ``face``  — a person photo. URL-based providers get a temp-hosted crop
                when a public base URL is configured; otherwise keyless
                metadata + local face verification of candidate thumbnails.
  - ``image`` — generic reverse-image search (product/scene/meme). URL-based
                providers get the image URL directly, or a temp-hosted upload.

Every provider reports its own enablement honestly via ``is_enabled()`` and
``note()`` so the orchestrator can explain *why* a source did not run.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..models import ProviderResult, RawMatch


class Provider(ABC):
    """One searchable source. Subclasses set ``name`` + capability flags."""

    name: str = ""
    accepts_url: bool = False  # can consume a publicly reachable image URL
    accepts_upload: bool = False  # can consume raw image bytes directly
    needs_face: bool = False  # meaningful only for person photos

    def is_enabled(self) -> bool:
        """Whether this provider has the credentials/deps it needs."""
        return True

    def note(self) -> Optional[str]:
        """Why the provider is disabled / any caveats (None = no caveats)."""
        return None

    @abstractmethod
    def search(
        self,
        client: Any,  # httpx.Client (sync)
        *,
        image_url: Optional[str],
        image_bytes: Optional[bytes],
        image_filename: Optional[str],
        max_results: int,
    ) -> ProviderResult:
        """Run one query. Must never raise — return ProviderResult(error=...)."""

    # ── helpers for subclasses ────────────────────────────────────────

    @staticmethod
    def _result(
        provider: str,
        items: list[dict],
        max_results: int,
        url_keys: tuple[str, ...] = ("link", "url", "source"),
        title_keys: tuple[str, ...] = ("title", "name", "source"),
        thumb_keys: tuple[str, ...] = ("thumbnail", "image", "original_image"),
    ) -> ProviderResult:
        """Parse a provider-native list of dicts into RawMatches."""
        matches: list[RawMatch] = []
        for item in items[:max_results]:
            if not isinstance(item, dict):
                continue
            url = next((item[k] for k in url_keys if item.get(k)), None)
            if not url:
                continue
            matches.append(
                RawMatch(
                    url=str(url),
                    title=item.get(title_keys[0]) or next(
                        (item[k] for k in title_keys[1:] if item.get(k)), None
                    ),
                    thumbnail_url=item.get(thumb_keys[0]) or next(
                        (item[k] for k in thumb_keys[1:] if item.get(k)), None
                    ),
                    score=item.get("score"),
                    source=provider,
                )
            )
        return ProviderResult(provider=provider, matches=matches)

    @staticmethod
    def _get_json(client: Any, url: str, params: dict) -> dict:
        """GET + raise_for_status + json(). Lets httpx errors propagate to
        the provider's own try/except (which converts them to error results)."""
        resp = client.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
