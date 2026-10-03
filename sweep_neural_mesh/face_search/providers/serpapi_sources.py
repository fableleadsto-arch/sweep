"""SerpAPI-backed providers (one API key unlocks all four engines).

Engines: google_lens, yandex_images, bing_reverse_image, google_images.
All require a publicly reachable image URL (SerpAPI fetches it server-side).
Bing is marked experimental: Microsoft retired the official Bing Visual
Search API in 2025, so results come from SerpAPI scraping Bing's public UI.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from .base import Provider

SERPAPI_URL = "https://serpapi.com/search.json"


def serpapi_key() -> str:
    return os.environ.get("SERPAPI_KEY", "").strip()


class _SerpApiProvider(Provider):
    """Shared plumbing for the four SerpAPI engines."""

    engine: str = ""
    url_param: str = "url"  # bing uses image_url
    experimental: bool = False

    def is_enabled(self) -> bool:
        return bool(serpapi_key())

    def note(self) -> Optional[str]:
        if not self.is_enabled():
            return f"Set SERPAPI_KEY to enable {self.engine}."
        if self.experimental:
            return (
                "Experimental — upstream retired the official API; results come "
                "from scraping Bing's public UI and may break without warning."
            )
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
        if not self.is_enabled():
            return self._disabled()
        if not image_url:
            return self._needs_url()
        params: dict[str, str] = {
            "engine": self.engine,
            self.url_param: image_url,
            "api_key": serpapi_key(),
        }
        try:
            data = self._get_json(client, SERPAPI_URL, params)
        except Exception as exc:  # noqa: BLE001 — network/HTTP errors are data
            return self._error(exc)
        return self._result(
            self.name,
            self._extract(data)[:max_results],
            max_results,
        )

    def _extract(self, data: dict) -> list[dict]:
        raise NotImplementedError

    def _disabled(self) -> Any:
        from ..models import ProviderResult

        return ProviderResult(provider=self.name, error=f"SERPAPI_KEY not set ({self.engine})")

    def _needs_url(self) -> Any:
        from ..models import ProviderResult

        return ProviderResult(
            provider=self.name,
            error=f"{self.engine} requires a publicly reachable image URL",
        )

    def _error(self, exc: Exception) -> Any:
        from ..models import ProviderResult

        return ProviderResult(provider=self.name, error=str(exc)[:300])


class GoogleLensProvider(_SerpApiProvider):
    name = "google_lens"
    engine = "google_lens"
    accepts_url = True

    def _extract(self, data: dict) -> list[dict]:
        return list(data.get("visual_matches") or [])


class YandexImagesProvider(_SerpApiProvider):
    name = "yandex_images"
    engine = "yandex_images"
    accepts_url = True

    def _extract(self, data: dict) -> list[dict]:
        # yandex returns sites_with_image (pages) — richest for profiles
        return list(data.get("sites_with_image") or data.get("images_results") or [])


class BingReverseImageProvider(_SerpApiProvider):
    name = "bing_reverse_image"
    engine = "bing_reverse_image"
    url_param = "image_url"
    accepts_url = True
    experimental = True

    def _extract(self, data: dict) -> list[dict]:
        return list(
            data.get("pages_with_matching_images")
            or data.get("image_results")
            or data.get("visual_matches")
            or data.get("inline_images")
            or []
        )


class GoogleImagesProvider(_SerpApiProvider):
    """Classic Google reverse-image (openface-search's original source)."""

    name = "google_images"
    engine = "google_images"
    accepts_url = True

    def _extract(self, data: dict) -> list[dict]:
        return list(
            data.get("image_results")
            or data.get("inline_images")
            or data.get("visual_matches")
            or []
        )


SERPAPI_PROVIDERS: list[type[Provider]] = [
    GoogleLensProvider,
    YandexImagesProvider,
    BingReverseImageProvider,
    GoogleImagesProvider,
]
