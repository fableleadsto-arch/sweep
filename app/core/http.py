"""Bounded public-web fetching with pinned DNS, redirects, retries and caching."""
from __future__ import annotations

import asyncio
import random
import time
import weakref
from urllib.parse import urljoin, urlparse

import httpx

from .guard import assert_safe_url, resolve_public_addresses
from .types import FetchResult

USER_AGENTS = ["Sweep/2.0 (personal web research)"]
HOST_MIN_INTERVAL_MS = 700
MAX_CONCURRENT = 6
CACHE_TTL_MS = 120_000
CACHE_MAX_ENTRIES = 200
MAX_RESPONSE_BYTES = 10_000_000
_cache: dict[tuple, tuple[float, FetchResult]] = {}
# Each event loop owns its primitives (the CLI uses asyncio.run repeatedly).
_limits = weakref.WeakKeyDictionary()


def _limiter():
    loop = asyncio.get_running_loop()
    if loop not in _limits:
        _limits[loop] = (asyncio.Semaphore(MAX_CONCURRENT), {})
    return _limits[loop]


async def _polite_delay(host: str) -> None:
    _, schedule = _limiter()
    now = time.monotonic()
    # Reservation happens before awaiting, so same-host requests cannot race.
    scheduled = max(now, schedule.get(host, now))
    schedule[host] = scheduled + HOST_MIN_INTERVAL_MS / 1000
    if len(schedule) > 1000:
        for key, due in list(schedule.items()):
            if due < now:
                schedule.pop(key, None)
    await asyncio.sleep(max(0, scheduled - now))


def _cache_get(key):
    entry = _cache.get(key)
    if entry and time.monotonic() - entry[0] <= CACHE_TTL_MS / 1000:
        return entry[1].model_copy(deep=True)
    _cache.pop(key, None)
    return None


def _cache_set(key, value):
    if len(_cache) >= CACHE_MAX_ENTRIES:
        _cache.pop(min(_cache, key=lambda k: _cache[k][0]), None)
    _cache[key] = (time.monotonic(), value.model_copy(deep=True))


CAPTCHA_MARKERS = ["captcha", "are you a robot", "unusual traffic", "verify you are human",
                   "cf-browser-verification", "checking your browser before", "access denied",
                   "enable javascript and cookies to continue"]


def looks_blocked(status: int, text: str) -> bool:
    return status in (403, 429, 503) or (
        len(text) >= 200 and any(marker in text[:4000].lower() for marker in CAPTCHA_MARKERS)
    )


def _headers_for(attempt: int) -> dict[str, str]:
    return {"User-Agent": USER_AGENTS[0], "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9", "Accept-Encoding": "identity"}


async def _read_public(client, url, method, headers, body, max_bytes):
    addresses = await asyncio.to_thread(resolve_public_addresses, url)
    original = httpx.URL(url)
    # Connect to the validated IP; preserve HTTP Host and TLS certificate/SNI hostname.
    target = original.copy_with(host=addresses[0])
    request_headers = {k: v for k, v in headers.items() if k.lower() not in {
        "host", "accept-encoding", "content-length", "transfer-encoding", "connection"
    }}
    request_headers["Host"] = original.netloc.decode("ascii")
    request_headers["Accept-Encoding"] = "identity"
    # The transport URL contains an IP, so an automatic cookie jar would confuse
    # different sites sharing that IP. This stateless gateway uses only explicit
    # caller cookies, which the redirect policy strips across origins.
    client.cookies.clear()
    async with client.stream(method, target, headers=request_headers, content=body,
                             extensions={"sni_hostname": original.host}) as response:
        if response.status_code in {301, 302, 303, 307, 308}:
            return response.status_code, dict(response.headers), b""
        # Raw streaming avoids unbounded decompression of a hostile compressed body.
        if response.headers.get("content-encoding", "identity").lower() not in {"", "identity"}:
            raise ValueError("Compressed response refused: server ignored Accept-Encoding: identity")
        content = bytearray()
        async for chunk in response.aiter_raw(chunk_size=65536):
            if len(content) + len(chunk) > max_bytes:
                raise ValueError(f"Response exceeds {max_bytes} bytes")
            content.extend(chunk)
        return response.status_code, dict(response.headers), bytes(content)


async def relai_fetch(raw_url: str, *, timeout_ms: int = 15_000, retries: int = 2,
                      method: str = "GET", body: str | None = None,
                      headers: dict[str, str] | None = None, cache: bool = True,
                      max_bytes: int = 2_000_000) -> FetchResult:
    """Fetch public HTTP(S); failures return a structured result, never cached."""
    def result(**kwargs):
        return FetchResult(url=raw_url, fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           **kwargs)
    try:
        url = assert_safe_url(raw_url)
        method = method.upper()
        if method not in {"GET", "POST"}:
            raise ValueError("Only GET and POST are supported")
        if not 1 <= max_bytes <= MAX_RESPONSE_BYTES:
            raise ValueError(f"max_bytes must be between 1 and {MAX_RESPONSE_BYTES}")
    except ValueError as exc:
        return result(ok=False, status=0, error=str(exc))
    # Credential/custom-header responses must never share a public cache.
    cacheable = cache and method == "GET" and not headers
    key = (url, max_bytes)
    if cacheable:
        cached = _cache_get(key)
        if cached:
            return cached
    retries = min(max(retries, 0), 4) if method == "GET" else 0
    timeout_s = min(max(timeout_ms / 1000, 0.1), 120)
    last = result(ok=False, status=0, error="Not attempted")
    # Generic proxy endpoints can fetch private URLs on our behalf, defeating pinning.
    # The configured legacy proxy is deliberately not used by this public-web gateway.
    async with httpx.AsyncClient(follow_redirects=False, trust_env=False,
                                timeout=timeout_s) as client:
        for attempt in range(retries + 1):
            current = url
            request_method, request_body = method, body
            request_headers = {**_headers_for(attempt), **(headers or {})}
            try:
                semaphore, _ = _limiter()
                async with semaphore, asyncio.timeout(timeout_s):
                    for redirect in range(5):
                        assert_safe_url(current)
                        await _polite_delay(urlparse(current).hostname or "")
                        status, response_headers, content = await _read_public(
                            client, current, request_method, request_headers, request_body, max_bytes,
                        )
                        if status in {301, 302, 303, 307, 308}:
                            location = response_headers.get("location")
                            if not location or redirect == 4:
                                raise ValueError("Missing redirect target or too many redirects")
                            following = assert_safe_url(urljoin(current, location))
                            previous_url, next_url = httpx.URL(current), httpx.URL(following)
                            if (previous_url.scheme, previous_url.host, previous_url.port) != (
                                    next_url.scheme, next_url.host, next_url.port):
                                request_headers = {k: v for k, v in request_headers.items()
                                                   if k.lower() not in {"authorization", "cookie", "proxy-authorization"}}
                                # Do not forward a sensitive POST body to a different origin.
                                if request_method != "GET":
                                    raise ValueError("Cross-origin POST redirect refused")
                            if status == 303 or (status in {301, 302} and request_method == "POST"):
                                request_method, request_body = "GET", None
                            current = following
                            continue
                        text = content.decode("utf-8", errors="replace")
                        blocked = looks_blocked(status, text)
                        last = FetchResult(ok=200 <= status < 300 and not blocked, status=status,
                                           url=current, text=text,
                                           content_type=response_headers.get("content-type", ""),
                                           blocked=blocked, attempts=attempt + 1,
                                           error="Blocked or challenged by host" if blocked else None,
                                           fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                        break
            except ValueError as exc:
                return result(ok=False, status=0, error=str(exc), attempts=attempt + 1)
            except (httpx.HTTPError, TimeoutError, OSError) as exc:
                last = result(ok=False, status=0, error=f"Fetch failed: {type(exc).__name__}",
                              attempts=attempt + 1)
            if last.ok:
                if cacheable:
                    _cache_set(key, last)
                return last
            if 400 <= last.status < 500 and last.status != 429:
                return last
            if attempt < retries:
                await asyncio.sleep(0.5 * 2 ** attempt + random.random() * 0.4)
    return last


async def pooled(items: list, limit: int, worker) -> list:
    """Run a bounded number of workers without creating one task per input."""
    size = min(max(limit, 1), MAX_CONCURRENT)
    results = [None] * len(items)
    indexed = iter(enumerate(items))

    async def run_worker():
        for index, item in indexed:
            results[index] = await worker(item, index)

    await asyncio.gather(*(run_worker() for _ in range(min(size, len(items)))))
    return results
