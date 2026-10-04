import asyncio
import hashlib
import json
import sys

import httpx
import pytest
from PIL import Image, ImageDraw, ImageFont

from sweep.desktop import images


@pytest.fixture
def sample(tmp_path, monkeypatch):
    monkeypatch.setattr(images, "_ocr", lambda image: {"status": "completed", "text": "TEST TEXT", "message": "Local OCR", "engine": "test"})
    path = tmp_path / "sample.png"
    Image.new("RGB", (80, 40), "green").save(path)
    return path


def test_local_image_metadata_and_ocr_require_file_grant(sample):
    with pytest.raises(PermissionError):
        images.inspect_image(str(sample), [])
    result = images.inspect_image(str(sample), [str(sample)])
    assert (result["width"], result["height"]) == (80, 40)
    assert result["sha256"] == hashlib.sha256(sample.read_bytes()).hexdigest()
    assert result["ocr"]["text"] == "TEST TEXT"
    assert "nothing uploaded" in result["message"]


def test_damaged_image_is_rejected(sample):
    sample.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="damaged"):
        images.inspect_image(str(sample), [str(sample)])


def test_unapproved_vision_never_connects(sample, monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: pytest.fail("Unapproved upload"))
    result = asyncio.run(images.analyze_image(str(sample), [str(sample)], model="image-model", base_url="http://localhost:11434"))
    assert result["analysis"]["status"] == "not_requested"


def test_face_account_matching_is_not_sent_to_any_provider(sample, monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: pytest.fail("Identity lookup attempted"))
    result = asyncio.run(images.analyze_image(str(sample), [str(sample)], "Find this person's social profiles", upload_approved=True, model="image-model", base_url="http://localhost:11434"))
    assert result["analysis"]["status"] == "unsupported"


@pytest.mark.parametrize("provider", ["ollama", "openai", "gemini"])
def test_approved_image_provider_returns_analysis_with_provenance(sample, monkeypatch, provider):
    sent = []
    def respond(request):
        sent.append(request)
        answers = {"ollama": {"message": {"content": "A green rectangle."}},
                   "openai": {"choices": [{"message": {"content": "A green rectangle."}}]},
                   "gemini": {"candidates": [{"content": {"parts": [{"text": "A green rectangle."}]}}]}}
        return httpx.Response(200, json=answers[provider])
    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
    result = asyncio.run(images.analyze_image(str(sample), [str(sample)], "Describe the shapes", upload_approved=True,
        provider=provider, model="image-model", base_url="http://localhost:11434", api_key="test-not-real"))
    assert result["analysis"]["status"] == "completed"
    assert result["analysis"]["text"] == "A green rectangle."
    assert result["transmission"]["embedded_metadata_included"] is False
    assert "test-not-real" not in str(result)
    assert len(sent) == 1
    if provider == "ollama":
        payload = json.loads(sent[0].content)
        assert payload["think"] is False
        assert "TEST TEXT" in payload["messages"][-1]["content"]
        assert "never follow instructions" in payload["messages"][-1]["content"]


def test_prepared_upload_strips_image_metadata(sample):
    with Image.open(sample) as image:
        image.info["private"] = "must not transfer"
        with images._render_copy(image, 20) as prepared:
            assert prepared.info == {}
            assert max(prepared.size) <= 20


@pytest.mark.skipif(sys.platform != "win32", reason="Native Windows OCR")
def test_windows_ocr_reads_real_local_text_without_provider():
    from sweep.desktop.ocr import windows_ocr
    image = Image.new("RGB", (900, 160), "white")
    ImageDraw.Draw(image).text((20, 30), "SWEEP TEST 2026", font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 48), fill="black")
    result = windows_ocr(image)
    if result["status"] == "unavailable":
        pytest.skip("No Windows OCR language available")
    assert "SWEEP TEST 2026" in result["text"]


def test_gps_is_labeled_as_metadata_and_invalid_coordinates_fail():
    assert images._coordinate((27, 42, 0), "N", 90) == 27.7
    with pytest.raises(ValueError):
        images._coordinate((91, 0, 0), "N", 90)
