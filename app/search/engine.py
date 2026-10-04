"""Multi-engine web search with HTML parsers and API provider fallback.

Provider list (in default routing order):
  keyless  — Multi-engine HTML search (no key needed; providers can block requests)
  tavily   — when TAVILY_API_KEY is set
  exa      — when EXA_API_KEY is set
  searxng  — when SEARXNG_BASE_URL is set
  jina     — when JINA_API_KEY is set
"""

from __future__ import annotations

import asyncio
import base64
import binascii
from datetime import datetime, timedelta, timezone
from html import unescape
import json
import re
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

from bs4 import BeautifulSoup

from ..core.guard import validate_safe_url
from ..core.http import relai_fetch


# ── Engine Parsers ────────────────────────────────────────────────────


def _first_match(text: str, patterns: list[re.Pattern]) -> str:
    """Try all patterns and return the first non-empty match group."""
    for pat in patterns:
        m = pat.search(text)
        if m and m.group(1):
            return _strip_html(m.group(1))
    return ""


def _strip_html(text: str) -> str:
    return unescape(re.sub(r"<[^>]+>", "", text)).strip()


def _decode_redirect(href: str) -> str:
    """Extract the actual destination, never a search engine's displayed citation."""
    href = unescape(href)
    try:
        parsed = urlparse(href if not href.startswith("//") else f"https:{href}")
    except ValueError:
        return ""
    host = (parsed.hostname or "").lower()
    if host in {"duckduckgo.com", "www.duckduckgo.com"} and parsed.path == "/l/":
        try:
            qs = parse_qs(parsed.query)
            return qs.get("uddg", [href])[0]
        except Exception:
            pass
    if host in {"bing.com", "www.bing.com"} and parsed.path == "/ck/a":
        target = parse_qs(parsed.query).get("u", [""])[0]
        if target.startswith("a1"):
            encoded = target[2:]
            try:
                return base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode("utf-8")
            except (ValueError, UnicodeDecodeError, binascii.Error):
                return ""
        return target if target.startswith(("https://", "http://")) else ""
    if href.startswith("/url?q="):
        try:
            return unquote(href[7].split("&")[0])
        except Exception:
            pass
    return href


def _parse_ddg(html: str) -> list[dict]:
    """Parse public HTML results independent of anchor attribute order."""
    document = BeautifulSoup(html, "html.parser")
    results = []
    for anchor in document.select("a.result__a[href]"):
        url = _decode_redirect(str(anchor.get("href", "")))
        title = anchor.get_text(" ", strip=True)
        block = anchor.find_parent(class_="result")
        snippet = block.select_one(".result__snippet") if block else None
        if validate_safe_url(url)[0] and title:
            results.append({"url": url, "title": title,
                            "snippet": snippet.get_text(" ", strip=True) if snippet else "",
                            "engine": "duckduckgo"})
    return results


def _parse_ddg_lite(html: str) -> list[dict]:
    """Parse Lite results, including snippets placed in the following table row."""
    document = BeautifulSoup(html, "html.parser")
    results = []
    for anchor in document.select("a.result-link[href]"):
        url = _decode_redirect(str(anchor.get("href", "")))
        title = anchor.get_text(" ", strip=True)
        row = anchor.find_parent("tr")
        following = row.find_next_sibling("tr") if row else None
        snippet = following.select_one(".result-snippet") if following else None
        if validate_safe_url(url)[0] and title:
            results.append({"url": url, "title": title,
                            "snippet": snippet.get_text(" ", strip=True) if snippet else "",
                            "engine": "duckduckgo-lite"})
    return results


def _parse_bing(html: str) -> list[dict]:
    """Use result anchors; displayed citations can contain spaces and shortened paths."""
    results = []
    seen = set()
    document = BeautifulSoup(html, "html.parser")
    for block in document.select("li.b_algo, div.b_algo"):
        anchor = block.select_one("h2 a[href]")
        if anchor is None:
            continue
        url = _decode_redirect(str(anchor.get("href", "")))
        title = anchor.get_text(" ", strip=True)
        paragraph = block.select_one(".b_caption p, p")
        snippet = paragraph.get_text(" ", strip=True) if paragraph else ""
        if validate_safe_url(url)[0] and title and url not in seen:
            seen.add(url)
            results.append({"url": url, "title": title, "snippet": snippet, "engine": "bing"})

    return results


def _parse_brave(html: str) -> list[dict]:
    """Brave Search parser."""
    results = []
    seen = set()

    blocks = re.findall(
        r'<div[^>]*class="[^"]*(?:snippet|result|search-result)[^"]*"[^>]*>[\s\S]*?<a[^>]+href="https?://[^"]+"[^>]*>[\s\S]*?</a>[\s\S]*?</div>',
        html,
        re.IGNORECASE,
    )

    for block in blocks:
        url_m = re.search(r'<a[^>]+href="(https?://[^"]+)"', block, re.I)
        if not url_m:
            continue
        url = url_m.group(1)
        title_m = re.search(r'class="[^"]*title[^"]*"[^>]*>([\s\S]*?)</div>', block, re.I)
        title = _strip_html(title_m.group(1)) if title_m else ""
        snippet_m = re.search(r'class="[^"]*snippet[^"]*"[^>]*>([\s\S]*?)</div>', block, re.I)
        snippet = _strip_html(snippet_m.group(1)) if snippet_m else ""
        if url and title and url not in seen:
            seen.add(url)
            results.append({"url": url, "title": title, "snippet": snippet, "engine": "brave"})

    return results


def _parse_mojeek(html: str) -> list[dict]:
    """Mojeek parser."""
    results = []
    seen = set()

    blocks = re.findall(
        r'<div[^>]*class="[^"]*result[^"]*"[^>]*>[\s\S]*?</div>\s*</div>',
        html,
        re.IGNORECASE,
    )

    for block in blocks:
        url_m = re.search(r'<a[^>]+href="(https?://[^"]+)"', block, re.I)
        if not url_m:
            continue
        url = url_m.group(1)
        title_m = re.search(r'class="[^"]*ob[^"]*"[^>]*>([\s\S]*?)</a>', block, re.I)
        if not title_m:
            title_m = re.search(r'<a[^>]+href="https?://[^"]+"[^>]*>([\s\S]*?)</a>', block, re.I)
        title = _strip_html(title_m.group(1)) if title_m else ""
        snippet_m = re.search(r'<p[^>]*class="[^"]*s[^"]*"[^>]*>([\s\S]*?)</p>', block, re.I)
        snippet = _strip_html(snippet_m.group(1)) if snippet_m else ""
        if url and title and url not in seen:
            seen.add(url)
            results.append({"url": url, "title": title, "snippet": snippet, "engine": "mojeek"})

    return results


def _parse_google(html: str) -> list[dict]:
    """Google parser (non-JS fallback)."""
    results = []
    seen = set()

    blocks = re.findall(
        r'<div[^>]*class="[^"]*g[^"]*"[^>]*>[\s\S]*?</div>\s*</div>',
        html,
        re.IGNORECASE,
    )

    for block in blocks:
        url_m = re.search(r'<a[^>]+href="(/url\?q=[^"&]+)[^"]*"', block, re.I)
        if url_m:
            url = unquote(url_m.group(1)[7].split("&")[0])
        else:
            url_m2 = re.search(r'<a[^>]+href="(https?://[^"]+)"', block, re.I)
            url = url_m2.group(1) if url_m2 else ""

        title_m = re.search(r'<h3[^>]*>([\s\S]*?)</h3>', block, re.I)
        title = _strip_html(title_m.group(1)) if title_m else ""
        snippet_m = re.search(r'class="[^"]*(?:VwiC3b|BNeawe)[^"]*"[^>]*>([\s\S]*?)</div>', block, re.I)
        snippet = _strip_html(snippet_m.group(1)) if snippet_m else ""

        if url and title and "google.com" not in url and url not in seen:
            seen.add(url)
            results.append({"url": url, "title": title, "snippet": snippet, "engine": "google"})

    return results


# ── Engine Registry ───────────────────────────────────────────────────


@dataclass
class Engine:
    name: str
    build_url: callable
    parse: callable


ENGINES: list[Engine] = [
    Engine(
        name="bing",
        build_url=lambda q, page: f"https://www.bing.com/search?q={quote_plus(q)}&setlang=en&cc=US&first={(page - 1) * 10 + 1}",
        parse=_parse_bing,
    ),
    Engine(
        name="duckduckgo",
        build_url=lambda q, page: f"https://html.duckduckgo.com/html/?q={quote_plus(q)}&kl=wt-wt&s={(page - 1) * 10}",
        parse=_parse_ddg,
    ),
    Engine(
        name="duckduckgo-lite",
        build_url=lambda q, _page: f"https://lite.duckduckgo.com/lite/?q={quote_plus(q)}",
        parse=_parse_ddg_lite,
    ),
    Engine(
        name="brave",
        build_url=lambda q, page: f"https://search.brave.com/search?q={quote_plus(q)}&source=web&offset={(page - 1) * 10}",
        parse=_parse_brave,
    ),
    Engine(
        name="mojeek",
        build_url=lambda q, page: f"https://www.mojeek.com/search?q={quote_plus(q)}&page={page}",
        parse=_parse_mojeek,
    ),
    Engine(
        name="google",
        build_url=lambda q, page: f"https://www.google.com/search?q={quote_plus(q)}&hl=en&start={(page - 1) * 10}",
        parse=_parse_google,
    ),
]


# ── API Providers ─────────────────────────────────────────────────────


_circuit_broken_until: dict[str, float] = {}


async def _provider_json(name: str, url: str, *, headers: dict | None = None,
                         payload: dict | None = None) -> tuple[dict | None, str | None]:
    """All search providers share the bounded, public-address HTTP guard."""
    response = await relai_fetch(
        url, method="POST" if payload is not None else "GET",
        body=json.dumps(payload) if payload is not None else None,
        headers={"Accept": "application/json", **({"Content-Type": "application/json"} if payload is not None else {}),
                 **(headers or {})},
        timeout_ms=6000, retries=0, cache=False, max_bytes=2_000_000,
    )
    if not response.ok:
        if response.status in {401, 402}:
            return None, "authentication or account credits required"
        if response.status == 429:
            _circuit_broken_until[name] = time.monotonic() + 60
            return None, "rate limited; trying another provider"
        if response.blocked:
            return None, f"provider blocked or challenged the request (HTTP {response.status})"
        if response.status:
            return None, f"HTTP {response.status}"
        return None, "connection failed or the public-address safety check refused the endpoint"
    try:
        data = json.loads(response.text)
    except (ValueError, TypeError):
        return None, "provider returned invalid JSON"
    if not isinstance(data, dict):
        return None, "provider returned an invalid response"
    return data, None


def _api_hits(data: dict, key: str, name: str, limit: int, snippet_key: str) -> dict:
    rows = data.get(key)
    if not isinstance(rows, list):
        return {"hits": [], "error": "provider returned no result list"}
    return {"hits": [
        {"url": row.get("url", ""), "title": row.get("title", ""),
         "snippet": str(row.get(snippet_key) or "")[:1500], "engine": name}
        for row in rows[:limit] if isinstance(row, dict)
    ]}


async def _tavily_search(query: str, limit: int = 10) -> dict:
    from ..config import get_settings
    settings = get_settings()
    if not settings.tavily_api_key:
        return {"hits": [], "error": "TAVILY_API_KEY not set"}
    if _circuit_broken_until.get("tavily", 0) > time.monotonic():
        return {"hits": [], "error": "temporarily rate limited"}
    data, error = await _provider_json("tavily", "https://api.tavily.com/search", payload={
        "api_key": settings.tavily_api_key, "query": query, "max_results": min(limit, 20), "include_answer": False,
    })
    return {"hits": [], "error": error} if error else _api_hits(data, "results", "tavily", limit, "content")


async def _exa_search(query: str, limit: int = 10) -> dict:
    from ..config import get_settings
    settings = get_settings()
    if not settings.exa_api_key:
        return {"hits": [], "error": "EXA_API_KEY not set"}
    if _circuit_broken_until.get("exa", 0) > time.monotonic():
        return {"hits": [], "error": "temporarily rate limited"}
    data, error = await _provider_json("exa", "https://api.exa.ai/search",
        headers={"x-api-key": settings.exa_api_key}, payload={
            "query": query, "numResults": limit, "type": "neural", "contents": {"text": True},
        })
    return {"hits": [], "error": error} if error else _api_hits(data, "results", "exa", limit, "text")


async def _searxng_search(query: str, limit: int = 10) -> dict:
    from ..config import get_settings
    settings = get_settings()
    if not settings.searxng_base_url:
        return {"hits": [], "error": "SEARXNG_BASE_URL not set"}
    url = f"{settings.searxng_base_url.rstrip('/')}/search?q={quote_plus(query)}&format=json&categories=general"
    data, error = await _provider_json("searxng", url)
    return {"hits": [], "error": error} if error else _api_hits(data, "results", "searxng", limit, "content")


async def _jina_search(query: str, limit: int = 10) -> dict:
    from ..config import get_settings
    settings = get_settings()
    if not settings.jina_api_key:
        return {"hits": [], "error": "JINA_API_KEY not set"}
    data, error = await _provider_json("jina", f"https://s.jina.ai/{quote_plus(query)}",
        headers={"Authorization": f"Bearer {settings.jina_api_key}"})
    return {"hits": [], "error": error} if error else _api_hits(data, "data", "jina", limit, "content")


# ── Bing Rate Gate ────────────────────────────────────────────────────

BING_MIN_INTERVAL_MS = 1500
_last_bing_call: float = 0


async def _bing_gate() -> None:
    global _last_bing_call
    elapsed = (time.time() - _last_bing_call) * 1000
    if elapsed < BING_MIN_INTERVAL_MS:
        await asyncio.sleep((BING_MIN_INTERVAL_MS - elapsed) / 1000)
    _last_bing_call = time.time()


# ── Main Search Function ──────────────────────────────────────────────


SEARCH_TIMEOUT_SECONDS = 20.0
AGGREGATE_TIMEOUT_SECONDS = 45.0
PROVIDER_TIMEOUT_SECONDS = 6.0


def _usable_hits(rows: object, provider: str) -> list[dict]:
    """Normalize provider output to the desktop/API result contract."""
    if not isinstance(rows, list):
        return []
    hits = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("url"), str):
            continue
        url = _decode_redirect(row["url"].strip())
        if len(url) > 4096 or not validate_safe_url(url)[0]:
            continue
        title = _strip_html(str(row.get("title") or ""))[:500]
        if not title:
            title = urlparse(url).hostname or url
        hits.append({"url": url, "title": title,
                     "snippet": _strip_html(str(row.get("snippet") or ""))[:1500], "engine": provider})
    return hits


async def relai_search(
    query: str,
    *,
    limit: int = 10,
    site: Optional[str] = None,
    rank: bool = True,
    mode: Optional[str] = None,
    page: int = 1,
    aggregate: bool = False,
) -> dict:
    """Return genuine public results with bounded provider fallbacks and diagnostics."""
    if not query.strip():
        raise ValueError("Enter a search query first.")
    search_query = f"site:{site} {query.strip()}" if site else query.strip()
    limit = min(max(limit, 1), 60)
    page = max(page, 1)
    if mode == "news" and "after:" not in search_query and "before:" not in search_query:
        recent = datetime.now(timezone.utc) - timedelta(days=365)
        search_query += f" after:{recent.date().isoformat()}"

    all_hits: list[dict] = []
    seen: set[str] = set()
    errors: list[str] = []
    attempted: list[str] = []
    loop = asyncio.get_running_loop()
    budget = AGGREGATE_TIMEOUT_SECONDS if aggregate else SEARCH_TIMEOUT_SECONDS
    deadline = loop.time() + budget

    def record(name: str, result: dict):
        hits = _usable_hits(result.get("hits"), name)
        if result.get("error"):
            errors.append(f"{name}: {result['error']}")
        elif not hits:
            errors.append(f"{name}: no usable results returned")
        for hit in hits:
            key = hit["url"].split("#", 1)[0]
            if key not in seen:
                seen.add(key)
                all_hits.append(hit)

    def finish() -> dict:
        hits = all_hits[:limit]
        return {
            "hits": hits, "engine": hits[0]["engine"] if hits else "none",
            "query": search_query,
            # Existing route_search uses this legacy field to detect an empty result.
            "tried": len(hits), "attempts": len(attempted), "attempted_providers": attempted,
            "errors": errors, "status": "complete" if hits else "unavailable", "blocked": not hits,
            "message": (f"Found {len(hits)} sources." if hits else
                        "No search provider returned usable results. Try again or configure a search API provider."),
        }

    from ..config import get_settings
    settings = get_settings()
    providers = []
    for configured, name, search_fn in (
        (settings.tavily_api_key, "tavily", _tavily_search),
        (settings.exa_api_key, "exa", _exa_search),
        (settings.searxng_base_url, "searxng", _searxng_search),
        (settings.jina_api_key, "jina", _jina_search),
    ):
        if configured:
            providers.append((name, search_fn))

    # Reserve at least half the total budget for public HTML fallback. A slow or
    # misconfigured API must not prevent the keyless path from being attempted.
    api_deadline = loop.time() + budget / 2
    for name, search_fn in providers:
        remaining = min(deadline, api_deadline) - loop.time()
        if remaining <= 0:
            break
        attempted.append(name)
        try:
            result = await asyncio.wait_for(search_fn(search_query, limit),
                                            timeout=min(PROVIDER_TIMEOUT_SECONDS, remaining))
            record(name, result)
        except TimeoutError:
            errors.append(f"{name}: timed out")
        except Exception as exc:
            # Provider exception strings can include URLs or API credentials.
            errors.append(f"{name}: request failed ({type(exc).__name__})")
        if all_hits and not aggregate:
            return finish()

    async def html_search(engine: Engine) -> dict:
        if engine.name == "bing":
            await _bing_gate()
        response = await relai_fetch(engine.build_url(search_query, page),
                                     timeout_ms=int(PROVIDER_TIMEOUT_SECONDS * 1000),
                                     retries=0, cache=False, max_bytes=2_000_000)
        if not response.ok:
            if response.blocked:
                error = f"provider blocked or challenged the request (HTTP {response.status})"
            elif response.status:
                error = f"HTTP {response.status}"
            else:
                error = "connection failed or public-address safety check refused the endpoint"
            return {"hits": [], "error": error}
        return {"hits": engine.parse(response.text)}

    for engine in ENGINES:
        remaining = deadline - loop.time()
        if remaining <= 0:
            errors.append("Search time budget reached; remaining providers were not attempted")
            break
        attempted.append(engine.name)
        try:
            result = await asyncio.wait_for(html_search(engine),
                                            timeout=min(PROVIDER_TIMEOUT_SECONDS, remaining))
            record(engine.name, result)
        except TimeoutError:
            errors.append(f"{engine.name}: timed out")
        except Exception as exc:
            errors.append(f"{engine.name}: request failed ({type(exc).__name__})")
        if all_hits and not aggregate:
            break
    if not all_hits and not errors:
        errors.append("No search providers were available")
    return finish()
