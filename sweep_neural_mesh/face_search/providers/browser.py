"""Browser escalation for keyless engine flows (Sweep's own stack).

Modern engine UIs (Bing sbi, Yandex cbir, Google Lens) render results with
JavaScript, so a plain HTTP response can be structurally complete yet empty of
results. This module drives a real browser through the same flow a human uses:

    goto(engine page) → set file input to the query image → wait for result
    selectors → return the final DOM HTML for the provider's parser.

Backends, tried in order:
  1. local Playwright (sync API) — `pip install playwright && playwright
     install chromium` (declared integration: requirements-sweep.txt /
     sweep_core.integrations.scraping).
  2. local Selenium + webdriver-manager style fallback is NOT attempted
     (selenium needs matching driver binaries); Playwright covers it.

Returns the rendered HTML, or None when no browser stack is available — the
provider then reports an honest, actionable error instead of crashing.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger("sweep.face_search.browser")

_lock = threading.Lock()
_pw_available: Optional[bool] = None  # memoized probe


def browser_available() -> bool:
    """Cheap check: is the Playwright Python stack importable?"""
    global _pw_available
    if _pw_available is None:
        try:
            import importlib.util

            _pw_available = importlib.util.find_spec("playwright") is not None
        except (ImportError, ValueError):
            _pw_available = False
    return _pw_available


def note_if_missing() -> str:
    return (
        "browser escalation unavailable — install Playwright for the keyless "
        "upload flow: pip install playwright && playwright install chromium"
    )


def browser_upload_html(
    *,
    start_url: str,
    image_path: str,
    file_selector: str = 'input[type="file"]',
    wait_selector: str,
    wait_timeout_ms: int = 25_000,
    settle_ms: int = 1_500,
) -> Optional[str]:
    """Run the upload flow in a real browser and return the final page HTML.

    Never raises — returns None on any failure (missing stack, navigation
    problems, selector timeouts, challenges left on screen).
    """
    if not browser_available():
        return None
    with _lock:  # a single browser at a time keeps memory predictable
        return _run_playwright(
            start_url=start_url,
            image_path=image_path,
            file_selector=file_selector,
            wait_selector=wait_selector,
            wait_timeout_ms=wait_timeout_ms,
            settle_ms=settle_ms,
        )


def _run_playwright(
    *,
    start_url: str,
    image_path: str,
    file_selector: str,
    wait_selector: str,
    wait_timeout_ms: int,
    settle_ms: int,
) -> Optional[str]:
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except Exception as exc:  # noqa: BLE001
        logger.debug("playwright import failed: %s", exc)
        return None

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                ctx = browser.new_context(
                    locale="en-US",
                    viewport={"width": 1366, "height": 900},
                )
                page = ctx.new_page()
                page.goto(start_url, timeout=45_000, wait_until="domcontentloaded")
                # Upload the query image through the engine's own file input.
                file_input = page.wait_for_selector(file_selector, timeout=15_000,
                                                    state="attached")
                if file_input is None:
                    return None
                file_input.set_input_files(image_path)
                # Results (or a challenge) appear after the JS upload flow.
                try:
                    page.wait_for_selector(wait_selector, timeout=wait_timeout_ms)
                except PWTimeout:
                    # No results selector — either zero results or a challenge.
                    pass
                page.wait_for_timeout(settle_ms)
                return page.content()
            finally:
                browser.close()
    except Exception as exc:  # noqa: BLE001 — any browser failure is None
        logger.debug("browser upload flow failed: %s", exc)
        return None
