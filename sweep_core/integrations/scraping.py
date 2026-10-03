"""Web scraping engine integrations.

Engines are exposed through lazy handles so Sweep only pays the import
cost of an engine when it is actually used. ``availability()`` reports
which engines are installed without importing the heavy ones.

Each engine has a Sweep-original handle name that abstracts over the
underlying third-party library.
"""

from __future__ import annotations

from typing import Any

from sweep_core.integrations import _module_available

ENGINES = (
    "scrapling",
    "scrapy",
    "playwright",
    "selenium",
    "pydoll",
    "camoufox",
)


def availability() -> dict[str, Any]:
    return {
        engine: {"available": _module_available(_IMPORT_NAMES[engine])}
        for engine in ENGINES
    }


_IMPORT_NAMES = {
    "scrapling": "scrapling",
    "scrapy": "scrapy",
    "playwright": "playwright",
    "selenium": "selenium",
    "pydoll": "pydoll",
    "camoufox": "camoufox",
}


def _handle(engine: str) -> Any:
    if engine not in _IMPORT_NAMES:
        raise KeyError(f"unknown scraping engine: {engine}")
    module = __import__(_IMPORT_NAMES[engine], fromlist=["__name__"])
    return module


def adaptive_fetch_handle() -> Any:
    """Adaptive fetching framework — uses anti-bot detection bypass."""
    return _handle("scrapling")


def spider_handle() -> Any:
    """Foundational crawling framework — broad crawling support."""
    return _handle("scrapy")


def browser_automation_handle() -> Any:
    """Chromium/Firefox/WebKit automation — headless browser control."""
    return _handle("playwright")


def webdriver_handle() -> Any:
    """Classic WebDriver automation — standard browser automation."""
    return _handle("selenium")


def chrome_automation_handle() -> Any:
    """WebDriver-free Chrome automation — no WebDriver dependency."""
    return _handle("pydoll")


def antidetect_handle() -> Any:
    """Anti-detect browser — fingerprint-resistant browsing."""
    return _handle("camoufox")


def antidetect_fetch(url: str, *, timeout_ms: int = 60_000) -> dict[str, Any]:
    """Fetch through the anti-detect browser backend.

    Falls back here when plain headless Chromium gets challenged by
    anti-bot layers (HTTP 202/403/challenge pages).
    """
    from scrapling.fetchers import StealthyFetcher

    page = StealthyFetcher.fetch(url, headless=True, timeout=timeout_ms)
    status = int(getattr(page, "status", 0) or 0)
    raw = getattr(page, "html_content", None) or getattr(page, "body", None) or ""
    return {"engine": "camoufox-stealth", "url": url, "status": status, "content": str(raw)}


def browser_fetch(url: str, *, timeout_ms: int = 30_000) -> dict[str, Any]:
    """Fetch a URL with a headless browser (sync API)."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            response = page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            content = page.content()
            status = response.status if response else 0
        finally:
            browser.close()
    return {"engine": "playwright", "url": url, "status": status, "content": content}
