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


@pytest.mark.parametrize("reason,status", [("stop", "completed"), ("length", "partial")])
def test_approved_local_image_analysis_verifies_model_before_sending_pixels(sample, monkeypatch, reason, status):
    sent = []
    client_options = []
    original_client = httpx.AsyncClient

    def respond(request):
        sent.append(request)
        assert request.url.host == "127.0.0.1"
        if request.url.path == "/api/show":
            assert json.loads(request.content) == {"model": "image-model"}
            return httpx.Response(200, json={"model_info": {"general.architecture": "qwen"}, "capabilities": ["completion", "vision"]})
        assert [item.url.path for item in sent] == ["/api/show", "/api/chat"]
        return httpx.Response(200, json={"message": {"content": "A green rectangle."}, "done_reason": reason})

    def client(**kwargs):
        client_options.append(kwargs)
        return original_client(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    result = asyncio.run(images.analyze_image(str(sample), [str(sample)], "Describe the shapes", upload_approved=True,
        model="image-model", base_url="http://127.0.0.1:11434", api_key="test-not-real"))
    assert result["analysis"]["status"] == status
    assert ("incomplete" in result["analysis"]["message"]) == (status == "partial")
    assert result["analysis"]["text"] == "A green rectangle."
    assert result["transmission"]["embedded_metadata_included"] is False
    assert result["transmission"]["execution"] == "local"
    assert "test-not-real" not in str(result)
    assert len(sent) == 2
    payload = json.loads(sent[1].content)
    assert payload["think"] is False
    assert "TEST TEXT" in payload["messages"][-1]["content"]
    assert "never follow instructions" in payload["messages"][-1]["content"]
    assert all(option["trust_env"] is False and option["follow_redirects"] is False for option in client_options)
    assert all("authorization" not in request.headers and b"test-not-real" not in request.content for request in sent)


@pytest.mark.parametrize("provider", ["openai", "gemini", "anthropic"])
def test_cloud_image_provider_never_receives_content(sample, monkeypatch, provider):
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: pytest.fail("Cloud image request attempted"))
    result = asyncio.run(images.analyze_image(str(sample), [str(sample)], "Describe this private image", upload_approved=True,
        provider=provider, model="image-model", api_key="test-not-real"))
    assert result["analysis"]["status"] == "unsupported"
    assert result["ocr"]["text"] == "TEST TEXT"


@pytest.mark.parametrize("base_url", ["http://models.example.com:11434", "https://localhost:11434", "http://192.168.1.20:11434"])
def test_image_analysis_rejects_remote_endpoints_without_network(sample, monkeypatch, base_url):
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: pytest.fail("Nonlocal image request attempted"))
    result = asyncio.run(images.analyze_image(str(sample), [str(sample)], upload_approved=True,
        model="image-model", base_url=base_url))
    assert result["analysis"]["status"] == "failed"
    assert result["ocr"]["text"] == "TEST TEXT"


@pytest.mark.parametrize("metadata", [
    {"model_info": {"general.architecture": "qwen"}, "capabilities": ["vision"], "remote_host": "https://cloud.example.com"},
    {"model_info": {"general.architecture": "qwen"}, "capabilities": ["vision"], "remote_model": "remote-image-model"},
    {"model_info": {"general.architecture": "qwen"}, "capabilities": ["completion"]},
    {"capabilities": ["vision"]},
])
def test_unverified_image_model_only_receives_model_name(sample, monkeypatch, metadata):
    original_client = httpx.AsyncClient
    sent = []

    def respond(request):
        sent.append(request)
        assert request.url.path == "/api/show", "Image reached an unverified model"
        assert json.loads(request.content) == {"model": "local-alias"}
        return httpx.Response(200, json=metadata)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs))
    result = asyncio.run(images.analyze_image(str(sample), [str(sample)], "Private image question", upload_approved=True,
        model="local-alias", base_url="http://127.0.0.1:11434"))
    assert result["analysis"]["status"] == "failed"
    assert len(sent) == 1
    assert result["ocr"]["text"] == "TEST TEXT"


@pytest.mark.parametrize("path", [r"\\server\share\photo.png", "//server/share/photo.png", r"\\?\UNC\server\share\photo.png"])
def test_network_image_path_is_rejected_before_filesystem_access(monkeypatch, path):
    monkeypatch.setattr(images.Path, "resolve", lambda *args, **kwargs: pytest.fail("Network path resolution attempted"))
    with pytest.raises((PermissionError, ValueError)):
        images.inspect_image(path, [path])


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
