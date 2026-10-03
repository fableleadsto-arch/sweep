"""Keyless engine scrapers — upload the image directly, no API keys.

Two-tier flow per engine (same order a human would experience):

  1. HTTP fast path — multipart upload to the engine's web endpoint and parse
     the server-rendered results (works while engines ship results in HTML).
  2. Browser escalation — when the response is client-rendered or challenged,
     drive a real browser through the engine's own upload UI via Playwright
     (providers.browser) and parse the rendered DOM.

Engines and their parsers:
  - bing:        upload → results embed per-image metadata in ``iusc`` blobs
                 (``m='{json}'``, keys murl/purl/turl/t) or modern ``data-m``
                 attributes; browser flow: bing.com/images → file input.
  - yandex:      upload → results page embeds ``CbirSites`` JSON blobs
                 (Domains + Sites [{Url, Title, Thumb}]); browser flow:
                 yandex.com/images → file input (cbir-uploader).
  - google_lens: upload → results page (community-documented flow); parse
                 generic [title, url] pairs from Lens data blobs; browser
                 flow: images.google.com → file input.
  - tineye:      upload → ``.match`` anchors / ``backlink`` JSON entries.

Every provider reports challenges and upstream changes as data
(ProviderResult.error / note) — a degraded engine never crashes the scan, and
the rest of the fleet still runs. No API keys anywhere.
"""
from __future__ import annotations

import html as html_mod
import json
import logging
import re
import tempfile
from typing import Any, Optional

from .base import Provider
from .http import HttpLike, fetch_html_anti_detect

logger = logging.getLogger("sweep.face_search.keyless")

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def _esc(s: str) -> str:
    """Unescape HTML entities and \\/ sequences inside embedded JSON."""
    return s.replace("\\/", "/").replace("&amp;", "&").replace("&quot;", '"')


def _extract_json_object(text: str, open_brace_idx: int) -> Optional[dict]:
    """Parse the balanced {...} object starting at an index (string-aware)."""
    if open_brace_idx >= len(text) or text[open_brace_idx] != "{":
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(open_brace_idx, min(len(text), open_brace_idx + 200_000)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    blob = json.loads(_esc(text[open_brace_idx:i + 1]))
                except (json.JSONDecodeError, ValueError):
                    return None
                return blob if isinstance(blob, dict) else None
    return None


def _host_of(url: str) -> str:
    try:
        import urllib.parse

        return urllib.parse.urlparse(url).netloc.lower()
    except ValueError:
        return ""


def _looks_like_challenge(status: int, text: str) -> bool:
    from .http import CHALLENGE_MARKERS

    if status in (403, 429, 202, 503):
        return True
    low = (text or "")[:4000].lower()
    return any(m in low for m in CHALLENGE_MARKERS)


def _challenge(provider: str, what: str) -> "Any":
    from ..models import ProviderResult

    return ProviderResult(
        provider=provider,
        error=f"{what} served an anti-bot challenge or client-rendered results",
        note="keyless scraping is best-effort; browser escalation also failed",
    )


def _failed(provider: str, exc: Exception, what: str) -> "Any":
    from ..models import ProviderResult

    return ProviderResult(provider=provider, error=f"{what}: {exc}"[:300])


def _to_raw_matches(provider: str, parsed: list[dict]):
    """Convert parser dicts to RawMatch objects (ProviderResult contract)."""
    from ..models import RawMatch

    return [
        RawMatch(
            url=d["link"],
            title=d.get("title") or None,
            thumbnail_url=d.get("thumbnail") or None,
            score=d.get("score"),
            source=provider,
        )
        for d in parsed
        if d.get("link")
    ]


class _EngineUploadProvider(Provider):
    """Shared two-tier logic: HTTP fast path → browser escalation."""

    accepts_upload = True
    # browser flow config (subclass responsibilities)
    start_url: str = ""
    wait_selector: str = ""

    def search(self, client: Any, *, image_url: Optional[str],
               image_bytes: Optional[bytes], image_filename: Optional[str],
               max_results: int) -> Any:
        from ..models import ProviderResult

        if not image_bytes:
            return ProviderResult(provider=self.name, error="requires image bytes")

        # 1. HTTP fast path
        res, need_browser = self._http_attempt(image_bytes, image_filename, max_results)
        if not need_browser:
            return res

        # 2. Browser escalation (real upload UI; handles JS-rendered results)
        html = self._browser_attempt(image_bytes, image_filename)
        if html is None:
            from .browser import note_if_missing

            if res is not None and getattr(res, "error", None):
                res.note = note_if_missing()
                return res
            return _challenge(self.name, self.name)
        matches = _to_raw_matches(self.name, self._parse(html, max_results))
        if not matches:
            return _challenge(self.name, self.name)
        from ..models import ProviderResult

        return ProviderResult(provider=self.name, matches=matches[:max_results])

    # ── subclass hooks ────────────────────────────────────────────────

    def _http_attempt(
        self, image_bytes: bytes, image_filename: Optional[str], max_results: int
    ) -> tuple[Optional[Any], bool]:
        """Run the HTTP flow. Returns (result, need_browser)."""
        raise NotImplementedError

    def _parse(self, html_text: str, max_results: int) -> list[dict]:
        raise NotImplementedError

    # ── shared plumbing ───────────────────────────────────────────────

    def _browser_attempt(
        self, image_bytes: bytes, image_filename: Optional[str]
    ) -> Optional[str]:
        """Upload through the engine's own UI in a real browser."""
        from . import browser

        if not browser.browser_available():
            return None
        tmp = None
        try:
            tmp = tempfile.NamedTemporaryFile(
                prefix="sweep_fs_up_", suffix=".jpg", delete=False
            )
            tmp.write(image_bytes)
            tmp.close()
            return browser.browser_upload_html(
                start_url=self.start_url,
                image_path=tmp.name,
                wait_selector=self.wait_selector,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("%s browser flow failed: %s", self.name, exc)
            return None
        finally:
            if tmp is not None:
                try:
                    import os

                    os.unlink(tmp.name)
                except OSError:
                    pass


# ══════════════════════════════════════════════════════════════════════
# Bing
# ══════════════════════════════════════════════════════════════════════

_BING_UPLOAD_URL = "https://www.bing.com/images/search"


class BingUploadProvider(_EngineUploadProvider):
    name = "bing_upload"
    start_url = "https://www.bing.com/images"
    wait_selector = ".iusc, .b_no, .dg_b, #b_results"

    def _http_attempt(self, image_bytes, image_filename, max_results):
        session = HttpLike(timeout=40.0)
        try:
            files = {"imageBin": (image_filename or "query.jpg", image_bytes, "image/jpeg")}
            resp = session.post(
                _BING_UPLOAD_URL,
                data=None,
                files=files,
                params={"view": "detailv2", "iss": "sbi", "form": "SBIVSP"},
                headers={"Referer": "https://www.bing.com/"},
            )
            if _looks_like_challenge(resp.status_code, resp.text):
                return _challenge(self.name, "bing upload"), True
            resp.raise_for_status()
            matches = _to_raw_matches(self.name, self._parse(resp.text, max_results))
            from ..models import ProviderResult

            result = ProviderResult(
                provider=self.name,
                matches=matches[:max_results],
                note=None if matches else
                "bing served a client-rendered page (no inline results)",
            )
            return result, not matches  # escalate when nothing parsed
        except Exception as exc:  # noqa: BLE001
            return _failed(self.name, exc, "bing upload"), True
        finally:
            session.close()

    def _parse(self, html_text: str, max_results: int) -> list[dict]:
        out: list[dict] = []
        seen: set[str] = set()
        # classic iusc blobs: <a class="iusc" m="{...}">
        for m in re.finditer(r'class="iusc"[^>]*\sm="([^"]+)"', html_text):
            try:
                blob = json.loads(_esc(html_mod.unescape(m.group(1))))
            except (json.JSONDecodeError, ValueError):
                continue
            purl = blob.get("purl")
            if not purl or purl in seen:
                continue
            seen.add(purl)
            out.append({
                "link": purl,
                "title": blob.get("t") or "",
                "thumbnail": blob.get("turl") or "",
                "_original_image": blob.get("murl") or "",
            })
            if len(out) >= max_results:
                return out
        # modern data-m attributes: data-m="{murl:...,purl:...}"
        for m in re.finditer(r'data-m="([^"]+)"', html_text):
            try:
                blob = json.loads(_esc(html_mod.unescape(m.group(1))))
            except (json.JSONDecodeError, ValueError):
                continue
            purl = blob.get("purl") or blob.get("PageUrl")
            if not purl or purl in seen:
                continue
            seen.add(purl)
            out.append({
                "link": purl,
                "title": blob.get("t") or "",
                "thumbnail": blob.get("turl") or "",
                "_original_image": blob.get("murl") or blob.get("MediaUrl") or "",
            })
            if len(out) >= max_results:
                break
        return out


# ══════════════════════════════════════════════════════════════════════
# Yandex
# ══════════════════════════════════════════════════════════════════════

_YANDEX_UPLOAD = "https://yandex.com/images-apphost/image-download"


class YandexUploadProvider(_EngineUploadProvider):
    name = "yandex_upload"
    start_url = "https://yandex.com/images/"
    wait_selector = ".CbirSites, .SitesList, .cbir-sites, .MiniSites, .b-no-results"

    def _http_attempt(self, image_bytes, image_filename, max_results):
        session = HttpLike(timeout=40.0)
        try:
            files = {"upfile": (image_filename or "query.jpg", image_bytes, "image/jpeg")}
            resp = session.post(
                _YANDEX_UPLOAD,
                data=None,
                files=files,
                params={"cbird": "111", "images_avatars_size": "orig"},
                headers={"Referer": "https://yandex.com/images/"},
            )
            results_url = resp.headers.get("location") or self._extract_results_url(resp.text)
            if not results_url:
                if _looks_like_challenge(resp.status_code, resp.text):
                    return _challenge(self.name, "yandex upload"), True
                from ..models import ProviderResult

                return (
                    ProviderResult(
                        provider=self.name,
                        matches=[],
                        note="yandex upload endpoint declined the request (flow changed)",
                    ),
                    True,
                )
            page = session.get(results_url, headers={"Referer": "https://yandex.com/images/"})
            if _looks_like_challenge(page.status_code, page.text):
                html_text = fetch_html_anti_detect(results_url)
                if not html_text:
                    return _challenge(self.name, "yandex results"), True
            else:            page.raise_for_status()
            html_text = page.text
            matches = _to_raw_matches(self.name, self._parse(html_text, max_results))
            from ..models import ProviderResult

            return ProviderResult(provider=self.name, matches=matches[:max_results]), not matches
        except Exception as exc:  # noqa: BLE001
            return _failed(self.name, exc, "yandex upload"), True
        finally:
            session.close()

    @staticmethod
    def _extract_results_url(body: str) -> Optional[str]:
        m = re.search(r'https://yandex\.com/images/search\?[^"\'<>\s]+', body or "")
        return m.group(0).replace("&amp;", "&") if m else None

    def _parse(self, html_text: str, max_results: int) -> list[dict]:
        out: list[dict] = []
        seen: set[str] = set()
        for m in re.finditer(r'"CbirSites?"\s*:\s*', html_text):
            blob = _extract_json_object(html_text, m.end())
            if not blob:
                continue
            for site in blob.get("Sites") or []:
                if not isinstance(site, dict):
                    continue
                url = site.get("Url") or site.get("url")
                if not url or url in seen:
                    continue
                seen.add(url)
                out.append({
                    "link": url,
                    "title": site.get("Title") or site.get("title") or "",
                    "thumbnail": site.get("Thumb") or site.get("thumb") or "",
                })
                if len(out) >= max_results:
                    return out
        # browser-rendered DOM fallback: .CbirSites-Item / .SitesList anchors
        for m in re.finditer(
            r'<a[^>]+class="[^"]*(?:Link|serp-item__link)[^"]*"[^>]+href="(https?://[^"]+)"',
            html_text,
        ):
            url = _esc(m.group(1))
            if url.startswith("https://yandex"):
                continue
            if url in seen:
                continue
            seen.add(url)
            out.append({"link": url, "title": "", "thumbnail": ""})
            if len(out) >= max_results:
                break
        return out


# ══════════════════════════════════════════════════════════════════════
# Google Lens
# ══════════════════════════════════════════════════════════════════════


class GoogleLensUploadProvider(_EngineUploadProvider):
    name = "google_lens_upload"
    start_url = "https://images.google.com/"
    wait_selector = ".LC20lb, .QBQfYb, .UAeSwb, #search a[href^='http'], .VUf7mf"

    def _http_attempt(self, image_bytes, image_filename, max_results):
        session = HttpLike(timeout=40.0)
        try:
            session.get("https://images.google.com/")
            files = {"encoded_image": (image_filename or "query.jpg", image_bytes, "image/jpeg")}
            resp = session.post(
                "https://www.google.com/searchbyimage/upload",
                data=None,
                files=files,
                headers={"Referer": "https://images.google.com/"},
            )
            results_url = resp.headers.get("location") or str(resp.url)
            if "/sorry/" in results_url or _looks_like_challenge(resp.status_code, resp.text):
                return _challenge(self.name, "google upload"), True
            page = session.get(results_url)
            if "/sorry/" in str(page.url) or _looks_like_challenge(page.status_code, page.text):
                return _challenge(self.name, "google results"), True
            page.raise_for_status()
            matches = _to_raw_matches(self.name, self._parse(page.text, max_results))
            from ..models import ProviderResult

            return ProviderResult(provider=self.name, matches=matches[:max_results]), not matches
        except Exception as exc:  # noqa: BLE001
            return _failed(self.name, exc, "google lens upload"), True
        finally:
            session.close()

    def _parse(self, html_text: str, max_results: int) -> list[dict]:
        out: list[dict] = []
        seen: set[str] = set()
        for m in re.finditer(
            r'\["((?:[^"\\]|\\.){3,180})","(https?://[^"]{10,300})"', html_text
        ):
            title, url = m.group(1), m.group(2)
            host = _host_of(url)
            if (
                not host
                or host.endswith(("google.com", "gstatic.com", "googleapis.com",
                                  "ggpht.com", "googleusercontent.com"))
                or "schema.org" in url
                or "w3.org" in url
            ):
                continue
            if url in seen:
                continue
            seen.add(url)
            out.append({"link": url, "title": title, "thumbnail": ""})
            if len(out) >= max_results:
                break
        return out


# ══════════════════════════════════════════════════════════════════════
# TinEye
# ══════════════════════════════════════════════════════════════════════


class TinEyeUploadProvider(_EngineUploadProvider):
    name = "tineye_upload"
    start_url = "https://tineye.com/"
    wait_selector = ".match, .result, #results"

    def _http_attempt(self, image_bytes, image_filename, max_results):
        session = HttpLike(timeout=40.0)
        try:
            session.get("https://tineye.com/")
            files = {"image": (image_filename or "query.jpg", image_bytes, "image/jpeg")}
            resp = session.post(
                "https://tineye.com/search",
                data=None,
                files=files,
                headers={"Referer": "https://tineye.com/"},
            )
            if _looks_like_challenge(resp.status_code, resp.text):
                return _challenge(self.name, "tineye upload"), True
            resp.raise_for_status()
            matches = _to_raw_matches(self.name, self._parse(resp.text, max_results))
            from ..models import ProviderResult

            return ProviderResult(provider=self.name, matches=matches[:max_results]), not matches
        except Exception as exc:  # noqa: BLE001
            return _failed(self.name, exc, "tineye upload"), True
        finally:
            session.close()

    def _parse(self, html_text: str, max_results: int) -> list[dict]:
        out: list[dict] = []
        seen: set[str] = set()
        for m in re.finditer(
            r'<a[^>]+class="match"[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>',
            html_text, flags=re.DOTALL,
        ):
            url, title = m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip()
            if url in seen:
                continue
            seen.add(url)
            out.append({"link": url, "title": title[:200], "thumbnail": ""})
            if len(out) >= max_results:
                break
        if not out:
            for m in re.finditer(r'"backlink"\s*:\s*"(https?://[^"]+)"', html_text):
                url = _esc(m.group(1))
                if url not in seen:
                    seen.add(url)
                    out.append({"link": url, "title": "", "thumbnail": ""})
                    if len(out) >= max_results:
                        break
        return out


KEYLESS_ENGINE_PROVIDERS: list[type[Provider]] = [
    BingUploadProvider,
    YandexUploadProvider,
    GoogleLensUploadProvider,
    TinEyeUploadProvider,
]
