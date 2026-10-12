"""Media chat routing, permission boundaries and preservation of useful results."""
import asyncio
import json
import wave
from types import SimpleNamespace

import pytest

from sweep.desktop.chat import understand
from sweep.desktop.runtime import CAPABILITIES, execute, run_worker


@pytest.mark.parametrize("name,text,capability", [
    ("clip.mp4", "inspect this video", "media.inspect"),
    ("clip.mov", "show the keyframes", "media.inspect"),
    ("clip.webm", "describe the visible objects", "media.analyze"),
    ("clip.mp4", "inspect this video and explain what happens", "media.analyze"),
    ("clip.mp4", "describe what is happening in these frames", "media.analyze"),
    ("clip.mp4", "what can you see in these frames?", "media.analyze"),
    ("clip.mp4", "what objects are in these frames?", "media.analyze"),
    ("clip.mp4", "inspect this video and identify the objects", "media.analyze"),
    ("clip.mp4", "describe the metadata only", "media.inspect"),
    ("clip.mkv", "transcribe this video", "audio.transcribe"),
    ("clip.mp4", "what was said?", "audio.transcribe"),
    ("meeting.mp3", "summarize this recording", "audio.transcribe"),
    ("meeting.wav", "what is its duration?", "media.inspect"),
    ("meeting.flac", "write subtitles", "audio.transcribe"),
])
def test_chat_routes_media_without_external_services(name, text, capability):
    assert understand(text, [name]) == (capability, text)
    assert CAPABILITIES[capability].execution == "local"


@pytest.mark.parametrize("text", ["transcribe this recording", "describe this video", "summarize the attached audio"])
def test_missing_attachment_is_clarified(text):
    assert understand(text)[0] == "chat.clarify"


@pytest.mark.parametrize("capability", ["media.inspect", "media.analyze", "audio.transcribe", "speech.setup"])
def test_media_and_setup_require_dispatch_approval(capability):
    with pytest.raises(PermissionError):
        asyncio.run(execute({"capability": capability, "text": "test"}, lambda event: None))


@pytest.mark.parametrize("capability", ["media.inspect", "media.analyze", "audio.transcribe"])
def test_media_requires_attached_file(capability):
    with pytest.raises(PermissionError, match="Attach"):
        asyncio.run(execute({"capability": capability, "text": "test", "approved": True}, lambda event: None))


def test_real_audio_worker_does_not_need_inference_or_model_setup(tmp_path, monkeypatch):
    from sweep.desktop import media_setup
    monkeypatch.setattr(media_setup, "install_speech_model", lambda *a, **k: pytest.fail("Unexpected download"))
    source = tmp_path / "silence.wav"
    with wave.open(str(source), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * 16000)
    original = source.read_bytes()
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"capability": "media.inspect", "text": "inspect",
        "approved": True, "granted_paths": [str(source)]}), encoding="utf-8")
    events = tmp_path / "events.jsonl"
    assert run_worker(str(task), str(events)) == 0
    result = json.loads(events.read_text(encoding="utf-8").splitlines()[-1])["data"]
    assert result["duration_seconds"] == pytest.approx(1)
    assert result["execution"] == "local" and source.read_bytes() == original


@pytest.fixture
def speech_request(tmp_path, monkeypatch):
    monkeypatch.setenv("SWEEP_DESKTOP_DIR", str(tmp_path / "profile"))
    from sweep.desktop import speech
    monkeypatch.setattr(speech, "transcribe_media", lambda *a, **k: {
        "text": "Meeting Monday. " * 600, "message": "Recognized speech.",
        "artifacts": [{"path": "kept-transcript.txt"}], "transcription": {"status": "completed"}})
    return {"capability": "audio.transcribe", "text": "summarize this recording", "approved": True,
            "granted_paths": [str(tmp_path / "meeting.wav")], "_artifact_dir": str(tmp_path / "artifacts")}


def test_recording_question_is_grounded_in_bounded_untrusted_transcript(speech_request, monkeypatch):
    from sweep.desktop import providers
    observed = []
    async def generate(root, **kwargs):
        observed.append(kwargs)
        return SimpleNamespace(text="The meeting is Monday.")
    monkeypatch.setattr(providers, "generate_chat", generate)
    result = asyncio.run(execute(speech_request, lambda event: None))
    assert result["summary"] == "The meeting is Monday."
    assert "5,000" in result["summary_scope"]
    assert "untrusted" in observed[0]["system"]
    assert len(observed[0]["messages"][0]["content"]) < 5100
    assert len(result["text"]) > 5000


def test_long_question_cannot_crowd_out_transcript_evidence(speech_request, monkeypatch):
    from sweep.desktop import providers
    speech_request["text"] = "Summarize. " * 1000
    async def generate(root, **kwargs):
        content = kwargs["messages"][0]["content"]
        assert len(content) <= 6000
        assert content.endswith(("Meeting Monday. " * 600)[:5000])
        return SimpleNamespace(text="The meeting is Monday.")
    monkeypatch.setattr(providers, "generate_chat", generate)
    result = asyncio.run(execute(speech_request, lambda event: None))
    assert "5,000" in result["summary_scope"]


@pytest.mark.parametrize("error", [RuntimeError("Local processing is unavailable."), ValueError("Invalid local settings.")])
def test_recording_transcript_survives_local_answer_failure(speech_request, monkeypatch, error):
    from sweep.desktop import providers
    async def unavailable(*args, **kwargs):
        raise error
    monkeypatch.setattr(providers, "generate_chat", unavailable)
    result = asyncio.run(execute(speech_request, lambda event: None))
    assert result["artifacts"] and result["text"]
    assert "preserved" in result["message"]


def test_slow_optional_summary_preserves_transcript(speech_request, monkeypatch):
    from sweep.desktop import media_tasks, providers
    monkeypatch.setattr(media_tasks, "OPTIONAL_ANSWER_SECONDS", 0)
    async def slow(*args, **kwargs):
        await asyncio.sleep(1)
        pytest.fail("The optional answer should time out")
    monkeypatch.setattr(providers, "generate_chat", slow)
    result = asyncio.run(execute(speech_request, lambda event: None))
    assert result["artifacts"] and result["text"]
    assert "exceeded its available time" in result["message"]


@pytest.fixture
def video_request(tmp_path, monkeypatch):
    from sweep.desktop import media, providers
    monkeypatch.setenv("SWEEP_DESKTOP_DIR", str(tmp_path / "profile"))
    directory = tmp_path / "artifacts"
    directory.mkdir()
    sheet = directory / "sheet.png"
    sheet.write_bytes(b"preview fixture")
    monkeypatch.setattr(media, "inspect_media", lambda *a, **k: {
        "message": "Sampled frames.", "contact_sheet": {"path": str(sheet)}, "artifacts": [{"path": str(sheet)}]})
    monkeypatch.setattr(providers, "load_config", lambda root: {"vision_model": "local-test", "ollama_base_url": "http://127.0.0.1:11434"})
    return {"capability": "media.analyze", "text": "describe the person carrying a box", "approved": True,
            "granted_paths": [str(tmp_path / "clip.mp4")], "_artifact_dir": str(directory)}


def test_frames_survive_engine_startup_failure(video_request, monkeypatch):
    from sweep.desktop import providers
    async def unavailable(*args, **kwargs):
        raise RuntimeError("The local engine is unavailable.")
    monkeypatch.setattr(providers, "start_local_server", unavailable)
    result = asyncio.run(execute(video_request, lambda event: None))
    assert result["artifacts"] and result["contact_sheet"]
    assert "preserved" in result["message"]


def test_visual_description_discloses_sampling_and_keeps_local_boundary(video_request, monkeypatch):
    from sweep.desktop import images, providers
    async def start(*args):
        return {}
    seen = []
    async def analyze(path, grants, prompt, **kwargs):
        assert not images.identity_request(prompt), "Visible person descriptions must not be falsely rejected"
        seen.append((path, grants, prompt, kwargs))
        return {"analysis": {"status": "completed", "text": "A box is visible."}}
    monkeypatch.setattr(providers, "start_local_server", start)
    monkeypatch.setattr(images, "analyze_image", analyze)
    result = asyncio.run(execute(video_request, lambda event: None))
    assert "not full-video analysis" in result["message"]
    assert "not seen continuous motion or heard any audio" in seen[0][2]
    assert seen[0][0] in seen[0][1]
    assert seen[0][3]["provider"] == "ollama"


def test_video_identity_request_is_rejected_before_analysis(video_request):
    video_request["text"] = "identify this person and find their accounts"
    with pytest.raises(ValueError, match="cannot identify"):
        asyncio.run(execute(video_request, lambda event: None))


def test_incomplete_visual_answer_remains_labelled(video_request, monkeypatch):
    from sweep.desktop import images, providers
    async def start(*args):
        return {}
    async def analyze(*args, **kwargs):
        return {"analysis": {"status": "partial", "text": "A box...", "message": "The answer is incomplete."}}
    monkeypatch.setattr(providers, "start_local_server", start)
    monkeypatch.setattr(images, "analyze_image", analyze)
    result = asyncio.run(execute(video_request, lambda event: None))
    assert "incomplete" in result["message"] and result["analysis"]["status"] == "partial"


@pytest.fixture
def dock(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from sweep.desktop.dock import DockWindow
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("SWEEP_DESKTOP_DIR", str(tmp_path))
    window = DockWindow(tmp_path)
    yield window
    window.shutdown()
    window.deleteLater()
    app.processEvents()


def test_missing_speech_model_offers_setup_without_claiming_transcription(dock):
    task = {"id": "a" * 32, "capability": "audio.transcribe"}
    result = {"message": "Install speech support.", "transcription": {"status": "unavailable"}}
    message = dock._result_message(task, result)
    assert message["state_label"] == "Setup needed"
    assert any(action[0] == "Set up speech" for action in message["actions"])


def test_partial_transcript_is_labelled_and_not_repeated(dock):
    task = {"id": "a" * 32, "capability": "audio.transcribe"}
    result = {"kind": "transcript", "message": "Partial transcript.\n\nMeeting Monday.",
              "text": "Meeting Monday.", "transcription": {"status": "partial"}}
    message = dock._result_message(task, result)
    assert message["state_label"] == "Partial transcript"
    assert message["text"].count("Meeting Monday.") == 1


def test_speech_setup_never_inherits_public_web_permission(dock, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    dock.settings.setValue("allow_web", True)
    seen = []
    def reject(dialog):
        seen.append(dialog.text())
        return QMessageBox.StandardButton.No
    monkeypatch.setattr(QMessageBox, "exec", reject)
    assert not dock.approve_task("speech.setup", "install speech")
    assert seen and "40 MB" in seen[0] and "alphacephei.com" in seen[0]


def test_contact_sheet_preview_stays_bound_to_task_and_cached(dock, tmp_path):
    from PIL import Image
    task = {"id": "a" * 32, "capability": "media.inspect"}
    directory = dock.task_store.directory(task["id"]) / "artifacts"
    directory.mkdir(parents=True)
    sheet = directory / "sheet.png"
    Image.new("RGB", (320, 180), "green").save(sheet)
    result = {"message": "Sampled frames.", "contact_sheet": {"path": str(sheet)}}
    first = dock._result_message(task, result)["preview"]
    assert dock._result_message(task, result)["preview"].cacheKey() == first.cacheKey()
    outside = tmp_path / "unrelated.png"
    Image.new("RGB", (80, 80), "red").save(outside)
    result["contact_sheet"]["path"] = str(outside)
    assert "preview" not in dock._result_message(task, result)


def test_video_overview_files_preserve_original_artifact_targets(dock):
    task = {"id": "a" * 32, "capability": "media.inspect"}
    result = {"message": "Sampled frames.", "contact_sheet": {"path": "unavailable"},
              "artifacts": [{"name": "overview.png", "format": "png"},
                            {"name": "frame.jpg", "format": "jpg"},
                            {"name": "timing.json", "format": "json"}]}
    files = dock._result_message(task, result)["artifacts"]
    assert [item["name"] for item in files] == ["overview.png", "timing.json", "frame.jpg"]
    assert [item["url"].rsplit("/", 1)[1] for item in files] == ["0", "2", "1"]
