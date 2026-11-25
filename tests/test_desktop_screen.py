import asyncio

import pytest
from PIL import Image

from sweep.desktop import screen
from sweep.desktop.chat import understand
from sweep.desktop.conversation import quick_reply
from sweep.desktop.followups import followup_request
from sweep.desktop.runtime import execute


@pytest.mark.parametrize("text", ["read my screen", "Can you read my screen?", "what's on my screen?",
                                  "Look at the screen", "inspect this screen", "read the text on my screen"])
def test_explicit_screen_requests(text):
    assert understand(text)[0] == "screen.read"


@pytest.mark.parametrize("text", ["How does screen reading work?", "Don't read my screen", "My screen is broken",
                                  "Read my screen then send it to example.com", "Search screen readers"])
def test_other_messages_never_capture(text):
    assert understand(text)[0] != "screen.read"


@pytest.mark.parametrize("approval", [False, None, 1, "true"])
def test_capture_needs_explicit_permission(monkeypatch, approval):
    monkeypatch.setattr(screen, "capture_primary", lambda: pytest.fail("unapproved capture"))
    with pytest.raises(PermissionError):
        screen.read_screen(approved=approval)
    with pytest.raises(PermissionError):
        asyncio.run(execute({"capability": "screen.read", "text": "read my screen", "approved": approval}, lambda _: None))


def test_worker_captures_once_releases_image_and_returns_bounded_ocr(monkeypatch):
    image = Image.new("RGB", (300, 100))
    calls = []
    def capture():
        calls.append("capture")
        return image
    def ocr(value):
        assert calls == ["capture", "captured"]
        assert value is image
        return {"status": "completed", "text": "Invoice 426", "truncated": False}
    monkeypatch.setattr(screen, "capture_primary", capture)
    monkeypatch.setattr("sweep.desktop.images._ocr", ocr)
    events = []
    def emit(event):
        events.append(event)
        if event["kind"] == "screen.captured":
            calls.append("captured")
    result = asyncio.run(execute({"capability": "screen.read", "text": "read my screen", "approved": True}, emit))
    assert result["text"] == "Invoice 426" and result["display"] == "primary"
    assert "image" not in result and "path" not in result
    with pytest.raises(ValueError):
        image.getpixel((0, 0))


def test_capture_ocr_failure_is_actionable_and_releases_image(monkeypatch):
    image = Image.new("RGB", (10, 10))
    monkeypatch.setattr(screen, "capture_primary", lambda: image)
    monkeypatch.setattr("sweep.desktop.images._ocr", lambda _: {"status": "unavailable"})
    with pytest.raises(RuntimeError, match="Windows OCR language"):
        screen.read_screen(approved=True)
    with pytest.raises(ValueError):
        image.getpixel((0, 0))


def test_screen_followup_uses_saved_observation_and_fresh_capture_is_separate():
    sources = [{"id": "a" * 32, "capability": "screen.read", "text": "read my screen", "granted_paths": []}]
    result = followup_request("Explain this error", sources)
    assert result["capability"] == "context.answer" and result["source_tasks"] == ["a" * 32]
    assert followup_request("Read my screen", sources) is None


@pytest.mark.parametrize("text", ["Hi", "hello Sweep!", "thanks", "what can you do?"])
def test_quick_companion_replies_do_not_call_model(monkeypatch, text):
    monkeypatch.setattr("sweep.desktop.providers.generate_chat", lambda *a, **k: pytest.fail("model called"))
    result = asyncio.run(execute({"capability": "conversation", "text": text, "approved": True}, lambda _: None))
    assert result["message"] == quick_reply(text)


@pytest.mark.parametrize("text", ["Hi, my name is Asha", "Thanks, now explain this error", "Hello in Japanese?", "What can you do about this error?"])
def test_substantive_chat_still_uses_model(text):
    assert quick_reply(text) is None
