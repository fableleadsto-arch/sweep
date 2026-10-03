"""Browse sessions — controlled browser layer.

A stateful browse session that keeps the system from ever touching raw
browser control: only controlled NavigationCommands are allowed
(goto/click/fill/scroll/back/forward/reload). Sessions are short-lived
in-memory handles.

Rendering path:
  - When Playwright/browserless WS is configured, initial page loads go
    through the browser service for JS-rendered content
  - Otherwise falls back to the resilient HTTP fetch layer
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Optional

import httpx

from ..core.http import relai_fetch
from ..core.types import (
    LinkData, NavigationCommand, NavigationResult, PageData,
    SearchAccessMode,
)
from ..core.guard import assert_safe_url
from ..extraction.page_data import extract_page_data

# ── Playwright Integration ────────────────────────────────────────────

_playwright_available = False
try:
    from playwright.async_api import async_playwright
    _playwright_available = True
except ImportError:
    pass


def playwright_configured() -> bool:
    """Check if a Playwright WebSocket endpoint is configured."""
    from ..config import get_settings
    settings = get_settings()
    return settings.allow_remote_browser and bool(settings.playwright_ws_endpoint or settings.browser_ws_endpoint)


async def execute_browser_action(action: dict) -> dict:
    """Execute a Playwright action via browser service or HTTP fallback."""
    from ..config import get_settings
    settings = get_settings()

    endpoint = settings.playwright_ws_endpoint or settings.browser_ws_endpoint
    if not settings.allow_remote_browser:
        return {"ok": False, "data": {}, "error": "Remote browser is disabled; use HTTP fetching"}
    if not endpoint:
        return {"ok": False, "data": {}, "error": "No browser endpoint configured"}

    if action.get("kind") == "navigate":
        return await _navigate_via_browser(endpoint, action.get("url", ""))

    return {"ok": False, "data": {}, "error": f"Action '{action.get('kind')}' requires a live browser session"}


async def _navigate_via_browser(endpoint: str, url: str) -> dict:
    """Navigate via browserless.io / Playwright Server HTTP API."""
    import json
    from ..core.guard import assert_safe_url, resolve_public_addresses

    url = assert_safe_url(url)
    await asyncio.to_thread(resolve_public_addresses, url)
    try:
        http_url = endpoint.replace("wss://", "https://").replace("ws://", "http://").rstrip("/")
        code = f"""
        const page = await browser.newPage();
        await page.setViewportSize({{ width: 1280, height: 800 }});
        await page.goto({json.dumps(url)}, {{ waitUntil: 'networkidle', timeout: 30000 }});
        const title = await page.title();
        const content = await page.evaluate(() => document.body.innerText);
        const links = await page.evaluate(() =>
          Array.from(document.querySelectorAll('a[href]'))
            .map(a => a.href)
            .filter(h => h.startsWith('http'))
        );
        const html = await page.content();
        await page.close();
        return {{ title, content: content.slice(0, 15000), links: links.slice(0, 50), html: html.slice(0, 10000) }};
        """
        async with asyncio.timeout(35):
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{http_url}/playwright",
                    json={"url": url, "code": code},
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "ok": True,
                        "data": {
                            "title": data.get("title", ""),
                            "url": url,
                            "text": data.get("content", ""),
                            "links": data.get("links", []),
                            "html": data.get("html", ""),
                        },
                    }
    except Exception:
        pass

    # Fallback to HTTP
    return await _fallback_http_fetch(url)


async def _fallback_http_fetch(url: str) -> dict:
    """HTTP fallback when browser service is unavailable."""
    result = await relai_fetch(url, retries=2, timeout_ms=15_000)
    if not result.ok:
        return {
            "ok": False,
            "data": {},
            "error": result.error or f"HTTP {result.status}",
        }

    # Extract title
    import re
    title_m = re.search(r"<title[^>]*>([^<]+)</title>", result.text, re.I)
    title = title_m.group(1) if title_m else ""

    # Extract links
    links = []
    for m in re.finditer(r'<a[^>]+href="(https?://[^"]+)"', result.text, re.I):
        if m.group(1) not in links:
            links.append(m.group(1))

    return {
        "ok": True,
        "data": {
            "title": title,
            "url": result.url,
            "text": result.text[:10_000],
            "links": links[:30],
        },
    }


# ── Session Management ────────────────────────────────────────────────

_sessions: dict[str, dict] = {}
_next_id = 0


def create_browse_session() -> dict:
    """Create a new browse session."""
    global _next_id
    session_id = f"browse_{int(time.time() * 1000).to_bytes(6, 'big').hex()}_{_next_id}"
    _next_id += 1
    _sessions[session_id] = {
        "id": session_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "url": "about:blank",
        "page": None,
        "back": [],
        "forward": [],
        "loading": False,
    }
    return {"id": session_id}


def get_browse_session(session_id: str) -> Optional[dict]:
    return _sessions.get(session_id)


def close_browse_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


def session_count() -> int:
    return len(_sessions)


# ── Navigation ────────────────────────────────────────────────────────

async def _load_page(url: str) -> dict:
    """Load a URL — browser service first, then HTTP."""
    safe_url = assert_safe_url(url)

    if playwright_configured():
        res = await execute_browser_action({"kind": "navigate", "url": safe_url})
        if res["ok"] and (res["data"].get("text") or res["data"].get("html")):
            page = extract_page_data(
                html=res["data"].get("html", "") or res["data"].get("text", ""),
                url=res["data"].get("url", safe_url),
                status=200,
                content_type="text/html",
                max_chars=20_000,
            )
            if res["data"].get("title"):
                page.title = res["data"]["title"]
            return {"page": page, "used_browser": True}

    # HTTP fallback
    result = await relai_fetch(safe_url, timeout_ms=15_000, retries=2)
    if not result.ok:
        return {
            "page": None,
            "used_browser": False,
            "blocked": result.blocked,
            "error": result.error or f"HTTP {result.status}",
        }

    is_json = "json" in result.content_type or re.match(r"^\s*[[{]", result.text[:200])
    page = extract_page_data(
        html=None if is_json else result.text,
        json_str=result.text if is_json else None,
        url=result.url,
        status=result.status,
        content_type=result.content_type,
        max_chars=20_000,
        fetched_at=result.fetched_at,
    )
    return {"page": page, "used_browser": False}


def _resolve_link(links: list[LinkData], target: str) -> Optional[LinkData]:
    """Resolve a click target to a real link on the page."""
    if not target:
        return None
    t = target.strip()
    t_lower = t.lower()

    # Exact URL match
    for link in links:
        if link.url == t or link.url.split("#")[0] == t.split("#")[0]:
            return link

    # Intent match
    for link in links:
        if link.intent and (link.intent == t_lower or t_lower in link.intent or link.intent in t_lower):
            return link

    # Anchor text match
    for link in links:
        if t_lower in link.text.lower():
            return link

    # Partial URL match
    for link in links:
        if t_lower in link.url.lower():
            return link

    return None


async def navigate(session_id: str, command: NavigationCommand) -> NavigationResult:
    """Execute a navigation command against a session."""
    session = _sessions.get(session_id)
    if not session:
        return NavigationResult(ok=False, url="", error="Browse session not found")
    if session["loading"]:
        return NavigationResult(ok=False, url=session["url"], error="A navigation is already in progress")

    kind = command.kind

    if kind == "goto":
        if not command.target:
            return NavigationResult(ok=False, url=session["url"], error="goto requires a target URL")
        prev = session["page"].url if session["page"] else None
        session["loading"] = True
        try:
            result = await _load_page(command.target)
            page = result.get("page")
            if not page:
                return NavigationResult(ok=False, url=command.target, blocked=result.get("blocked", False), error=result.get("error"))
            if prev and prev != page.url:
                session["back"].append(prev)
            session["forward"] = []
            session["url"] = page.url
            session["page"] = page
            return NavigationResult(
                ok=True, url=page.url, title=page.title,
                text=page.text[:12_000],
                links=page.links[:60],
                headings=page.headings[:40],
            )
        finally:
            session["loading"] = False

    elif kind == "back":
        if not session["back"]:
            return NavigationResult(ok=False, url=session["url"], error="No previous page")
        prev_url = session["back"].pop()
        result = await _load_page(prev_url)
        page = result.get("page")
        if not page:
            return NavigationResult(ok=False, url=prev_url, error=result.get("error"))
        if session["page"]:
            session["forward"].append(session["page"].url)
        session["url"] = page.url
        session["page"] = page
        return NavigationResult(ok=True, url=page.url, title=page.title, text=page.text[:12_000], links=page.links[:60])

    elif kind == "forward":
        if not session["forward"]:
            return NavigationResult(ok=False, url=session["url"], error="No forward page")
        next_url = session["forward"].pop()
        result = await _load_page(next_url)
        page = result.get("page")
        if not page:
            return NavigationResult(ok=False, url=next_url, error=result.get("error"))
        if session["page"]:
            session["back"].append(session["page"].url)
        session["url"] = page.url
        session["page"] = page
        return NavigationResult(ok=True, url=page.url, title=page.title, text=page.text[:12_000], links=page.links[:60])

    elif kind == "reload":
        result = await _load_page(session["url"])
        page = result.get("page")
        if not page:
            return NavigationResult(ok=False, url=session["url"], error=result.get("error"))
        session["page"] = page
        return NavigationResult(ok=True, url=page.url, title=page.title, text=page.text[:12_000], links=page.links[:60])

    elif kind == "click":
        if not session["page"]:
            return NavigationResult(ok=False, url=session["url"], error="No page loaded")
        link = _resolve_link(session["page"].links, command.target or "")
        if not link:
            return NavigationResult(ok=False, url=session["url"], error=f'No matching link for "{command.target}"')
        return await navigate(session_id, NavigationCommand(kind="goto", target=link.url))

    elif kind == "fill":
        if not playwright_configured():
            return NavigationResult(ok=False, url=session["url"], error="Fill requires a configured browser endpoint")
        if not command.selector:
            return NavigationResult(ok=False, url=session["url"], error="fill requires a selector")
        res = await execute_browser_action({"kind": "fill", "selector": command.selector, "value": command.value or ""})
        return NavigationResult(ok=res["ok"], url=session["url"], error=res.get("error"))

    elif kind == "scroll":
        if not playwright_configured():
            return NavigationResult(ok=True, url=session["url"], error="Scrolling requires a browser endpoint")
        res = await execute_browser_action({"kind": "scroll", "direction": command.direction or "down"})
        return NavigationResult(ok=res["ok"], url=session["url"], error=res.get("error"))

    return NavigationResult(ok=False, url=session["url"], error=f"Unsupported command: {kind}")


def current_page(session_id: str) -> Optional[PageData]:
    """Current page of a session."""
    session = _sessions.get(session_id)
    if not session:
        return None
    return session.get("page")
