"""One approved primary-screen capture, local OCR, and no retained screenshot."""
from __future__ import annotations

import os
import re
from datetime import UTC, datetime


def screen_request(text: str) -> bool:
    """Only explicit requests to look at the user's current screen trigger capture."""
    target = r"(?:my|the|this|current) screen"
    return bool(re.fullmatch(
        rf"(?:(?:read|inspect|scan|describe|look at|check|see) (?:the text (?:on|from) )?{target}"
        rf"|(?:what(?:'s| is)|what can you see) (?:shown |visible )?on {target})[.!?]*",
        text.strip(), re.I))


def capture_primary():
    if os.name != "nt":
        raise RuntimeError("Screen reading currently requires Windows.")
    # Workers have no Qt application to establish physical-pixel awareness.
    import ctypes

    from PIL import ImageGrab
    ctypes.windll.user32.SetProcessDPIAware()
    try:
        image = ImageGrab.grab(all_screens=False)
    except OSError:
        raise RuntimeError("The screen could not be captured. Unlock the desktop and try again.") from None
    if image.width * image.height > 40_000_000 or not image.width or not image.height:
        image.close()
        raise RuntimeError("The display is too large to read safely. Use a smaller display resolution.")
    return image


def read_screen(*, approved: bool, captured=None) -> dict:
    if approved is not True:
        raise PermissionError("Approve this screen capture before reading the screen.")
    from .images import _ocr
    image = capture_primary()
    try:
        timestamp = datetime.now(UTC).isoformat()
        width, height = image.size
        if captured:
            captured()
        ocr = _ocr(image)
    finally:
        image.close()
    if ocr.get("status") != "completed":
        raise RuntimeError("The screen was captured, but local text recognition could not run. "
                           "Install a Windows OCR language in Windows language settings and try again.")
    text = ocr.get("text", "").strip()
    message = ("I read the visible text on your primary display. "
               "This is a single capture; OCR can make mistakes. "
               "The screenshot was discarded. Recognized text stays in this chat.") if text else (
               "No readable text was found on the primary display. "
               "Try enlarging the text and reading the screen again. The screenshot was discarded.")
    return {"title": "Screen text", "kind": "screen", "message": message, "text": text,
            "captured_at": timestamp, "display": "primary", "width": width, "height": height,
            "truncated": ocr.get("truncated", False), "ocr": ocr,
            "scope": "Visible primary-display text only; no hidden windows, scrolling, or continuous monitoring."}
