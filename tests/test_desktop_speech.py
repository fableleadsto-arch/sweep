"""Offline speech boundaries with real media decoding and a deterministic recognizer."""

import asyncio
import hashlib
import json
import struct
import wave
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from sweep.desktop import speech

av = pytest.importorskip("av")
vosk = pytest.importorskip("vosk")


@pytest.fixture
def model_folder(tmp_path):
    folder = tmp_path / "speech-model"
    for relative in ("am/final.mdl", "conf/model.conf", "conf/mfcc.conf", "graph/phones/word_boundary.int", "graph/HCLr.fst", "graph/Gr.fst"):
        path = folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test model data")
    return folder


def wav_file(tmp_path, *, seconds=1, channels=1, rate=16000):
    path = tmp_path / "sample.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(struct.pack("<h", 100) * round(seconds * rate) * channels)
    return path


@pytest.fixture
def recognizer(monkeypatch):
    state = {"bytes": 0, "models": [], "words": False, "result": {
        "text": "hello sweep", "result": [
            {"word": "hello", "start": 0.1, "end": 0.3, "conf": 0.92},
            {"word": "sweep", "start": 0.3, "end": 0.7, "conf": 0.83},
        ],
    }}

    def model(*args, **kwargs):
        assert not args and set(kwargs) == {"model_path"}
        assert Path(kwargs["model_path"]).is_absolute()
        state["models"].append(kwargs["model_path"])
        return object()

    class Recognizer:
        def __init__(self, _model, sample_rate):
            assert sample_rate == 16000

        def SetWords(self, enabled):
            state["words"] = enabled

        def AcceptWaveform(self, payload):
            assert len(payload) % 2 == 0 and len(payload) <= 8000
            state["bytes"] += len(payload)
            return False

        def FinalResult(self):
            return json.dumps(state["result"])

    monkeypatch.setattr(vosk, "Model", model)
    monkeypatch.setattr(vosk, "KaldiRecognizer", Recognizer)
    return state


def transcribe(path, model_folder, tmp_path, **kwargs):
    return speech.transcribe_media(str(path), [str(path)], tmp_path / "artifacts", model_folder, **kwargs)


def test_real_wav_decode_produces_word_timing_and_three_reusable_artifacts(tmp_path, model_folder, recognizer):
    path = wav_file(tmp_path)
    original = path.read_bytes()
    result = transcribe(path, model_folder, tmp_path)
    assert result["text"] == "hello sweep"
    assert result["transcription"]["status"] == "completed"
    assert result["transcription"]["scope"]["processed_seconds"] == 1
    assert result["segments"][0] == {"text": "hello sweep", "start": 0.1, "end": 0.7}
    assert recognizer["words"] is True and recognizer["bytes"] == 32000
    assert recognizer["models"] == [str(model_folder.resolve())]
    assert [item["format"] for item in result["artifacts"]] == ["txt", "srt", "json"]
    artifacts = {item["format"]: Path(item["path"]) for item in result["artifacts"]}
    assert "00:00:00,100 --> 00:00:00,700" in artifacts["srt"].read_text()
    document = json.loads(artifacts["json"].read_text())
    assert document["segments"][0]["words"][0]["confidence"] == 0.92
    assert "Automatic transcription" in artifacts["txt"].read_text()
    for item in result["artifacts"]:
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]
    assert path.read_bytes() == original and result["sha256"] == hashlib.sha256(original).hexdigest()


def test_stereo_audio_resamples_to_mono_without_numpy(tmp_path, model_folder, recognizer):
    path = wav_file(tmp_path, channels=2, rate=44100)
    result = transcribe(path, model_folder, tmp_path)
    assert result["audio"]["channels"] == 2
    assert result["transcription"]["scope"]["processed_seconds"] == 1
    assert recognizer["bytes"] == 32000


@pytest.mark.parametrize("selection", [None, "", "relative/model", r"\\server\model", "https://example.com/model"])
def test_missing_or_nonlocal_model_preserves_source_metadata_without_loading(tmp_path, recognizer, selection):
    path = wav_file(tmp_path)
    result = transcribe(path, selection, tmp_path)
    assert result["transcription"]["status"] == "unavailable"
    assert "Open Settings and choose Set up speech" in result["message"]
    assert result["sha256"] and result["source_unchanged"]
    assert recognizer["models"] == [] and recognizer["bytes"] == 0
    assert result["artifacts"] == []


def test_incomplete_model_is_rejected_before_recognizer_load(tmp_path, model_folder, recognizer):
    (model_folder / "graph/Gr.fst").unlink()
    result = transcribe(wav_file(tmp_path), model_folder, tmp_path)
    assert result["transcription"]["status"] == "unavailable"
    assert recognizer["models"] == []


def test_single_graph_model_and_size_bounds(tmp_path, model_folder, monkeypatch):
    (model_folder / "graph/HCLr.fst").unlink()
    (model_folder / "graph/Gr.fst").unlink()
    (model_folder / "graph/HCLG.fst").write_bytes(b"single graph")
    assert speech.validate_model_path(model_folder) == model_folder.resolve()
    monkeypatch.setattr(speech, "MAX_MODEL_BYTES", 1)
    with pytest.raises(ValueError, match="model limit"):
        speech.validate_model_path(model_folder)


def test_model_traversal_count_is_bounded(model_folder, monkeypatch):
    monkeypatch.setattr(speech, "MAX_MODEL_FILES", 2)
    with pytest.raises(ValueError, match="too many files"):
        speech.validate_model_path(model_folder)


def test_exact_attachment_permission_precedes_speech_setup(tmp_path, model_folder, recognizer):
    path = wav_file(tmp_path)
    with pytest.raises(PermissionError, match="Attach this file"):
        speech.transcribe_media(str(path), [str(tmp_path)], tmp_path / "artifacts", model_folder)
    assert recognizer["models"] == []


def test_no_speech_is_a_clear_successful_result(tmp_path, model_folder, recognizer):
    recognizer["result"] = {"text": ""}
    result = transcribe(wav_file(tmp_path), model_folder, tmp_path)
    assert result["transcription"]["status"] == "no_speech"
    assert "No speech was recognized" in result["message"]
    assert result["segments"] == []
    assert len(result["artifacts"]) == 3


def test_audio_sample_limit_is_explicit_and_exact(tmp_path, model_folder, recognizer, monkeypatch):
    monkeypatch.setattr(speech, "MAX_AUDIO_SECONDS", 1)
    result = transcribe(wav_file(tmp_path, seconds=2), model_folder, tmp_path)
    scope = result["transcription"]["scope"]
    assert scope["truncated"] and scope["processed_seconds"] == 1
    assert result["transcription"]["status"] == "partial"
    assert recognizer["bytes"] == 32000
    assert "first five minutes" in result["message"]


def test_wall_clock_limit_stops_decoding(tmp_path, model_folder, recognizer, monkeypatch):
    monkeypatch.setattr(speech, "MAX_PROCESS_SECONDS", -1)
    recognizer["result"] = {"text": ""}
    result = transcribe(wav_file(tmp_path), model_folder, tmp_path)
    assert result["transcription"]["status"] == "partial"
    assert result["transcription"]["scope"]["processed_seconds"] == 0
    assert "processing limit" in result["message"]
    assert recognizer["bytes"] == 0


def fake_container(monkeypatch, *, positions, start_time=0, no_audio=False, decode_error=None):
    frames = []
    for position in positions:
        frame = av.AudioFrame(format="s16", layout="mono", samples=16000)
        frame.sample_rate = 16000
        frame.time_base = Fraction(1, 16000)
        frame.pts = None if position is None else round(position * 16000)
        frame.planes[0].update(bytes(32000))
        frames.append(frame)
    stream = SimpleNamespace(codec_context=SimpleNamespace(sample_rate=16000, channels=1, name="pcm_s16le"))

    class Container:
        streams = SimpleNamespace(audio=[] if no_audio else [stream])

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def decode(self, _stream):
            yield from frames
            if decode_error is not None:
                raise decode_error

    container = Container()
    container.start_time = start_time
    monkeypatch.setattr(speech, "open_media", lambda _raw: container)


@pytest.mark.parametrize("error", [av.error.InvalidDataError(1094995529, "damaged audio"), ValueError("invalid stream")])
def test_late_decode_failure_preserves_partial_transcript_and_artifacts(tmp_path, model_folder, recognizer, monkeypatch, error):
    fake_container(monkeypatch, positions=[0], decode_error=error)
    path = wav_file(tmp_path)
    original = path.read_bytes()
    result = transcribe(path, model_folder, tmp_path)
    assert result["transcription"]["status"] == "partial"
    assert result["transcription"]["scope"]["processed_seconds"] == 1
    assert result["text"] == "hello sweep"
    assert "Audio decoding stopped" in result["message"]
    artifacts = {item["format"]: Path(item["path"]) for item in result["artifacts"]}
    assert set(artifacts) == {"txt", "srt", "json"}
    assert "partial transcript" in artifacts["txt"].read_text()
    assert "hello sweep" in artifacts["srt"].read_text()
    document = json.loads(artifacts["json"].read_text())
    assert document["transcription"] == result["transcription"]
    assert document["text"] == result["text"]
    assert path.read_bytes() == original


@pytest.mark.parametrize("failure_point", ["second frame", "flush"])
def test_resampler_failure_retains_preceding_audio(tmp_path, model_folder, recognizer, monkeypatch, failure_point):
    fake_container(monkeypatch, positions=[0, 1] if failure_point == "second frame" else [0])
    original_resampler = av.AudioResampler

    class FailingResampler:
        def __init__(self, **kwargs):
            self.actual = original_resampler(**kwargs)
            self.calls = 0

        def resample(self, frame):
            self.calls += 1
            if self.calls == 2:
                raise ValueError("input audio format changed")
            return self.actual.resample(frame)

    monkeypatch.setattr(av, "AudioResampler", FailingResampler)
    result = transcribe(wav_file(tmp_path), model_folder, tmp_path)
    assert result["transcription"]["status"] == "partial"
    assert result["transcription"]["scope"]["processed_seconds"] == 1
    assert result["text"] == "hello sweep"
    assert "Audio conversion stopped" in result["message"]
    assert len(result["artifacts"]) == 3


def test_decode_failure_before_valid_audio_is_an_error(tmp_path, model_folder, recognizer, monkeypatch):
    fake_container(monkeypatch, positions=[], decode_error=av.error.InvalidDataError(1094995529, "damaged audio"))
    with pytest.raises(ValueError, match="Audio decoding could not begin"):
        transcribe(wav_file(tmp_path), model_folder, tmp_path)
    assert recognizer["bytes"] == 0
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("error", [RuntimeError("unexpected decoder bug"), asyncio.CancelledError()])
def test_unexpected_errors_and_cancellation_are_not_partial_success(tmp_path, model_folder, recognizer, monkeypatch, error):
    fake_container(monkeypatch, positions=[0], decode_error=error)
    with pytest.raises(type(error)):
        transcribe(wav_file(tmp_path), model_folder, tmp_path)
    assert not (tmp_path / "artifacts").exists()


def test_recognizer_value_error_is_not_misreported_as_bad_media(tmp_path, model_folder, recognizer, monkeypatch):
    class BrokenRecognizer(vosk.KaldiRecognizer):
        def AcceptWaveform(self, _payload):
            raise ValueError("unexpected recognizer bug")

    monkeypatch.setattr(vosk, "KaldiRecognizer", BrokenRecognizer)
    with pytest.raises(ValueError, match="unexpected recognizer bug"):
        transcribe(wav_file(tmp_path), model_folder, tmp_path)
    assert not (tmp_path / "artifacts").exists()


def test_audio_with_delayed_start_keeps_media_playback_offsets(tmp_path, model_folder, recognizer, monkeypatch):
    fake_container(monkeypatch, positions=[7, 8], start_time=2_000_000)
    result = transcribe(wav_file(tmp_path), model_folder, tmp_path)
    assert result["transcription"]["scope"]["timestamp_reference"] == "media_playback"
    assert result["transcription"]["scope"]["start_seconds"] == 5
    assert result["segments"][0]["start"] == 5.1
    assert result["segments"][0]["end"] == 5.7
    caption = next(item for item in result["artifacts"] if item["format"] == "srt")
    assert "00:00:05,100 --> 00:00:05,700" in Path(caption["path"]).read_text()


def test_discontinuous_timestamps_are_not_falsely_presented_as_video_time(tmp_path, model_folder, recognizer, monkeypatch):
    fake_container(monkeypatch, positions=[5, 9])
    result = transcribe(wav_file(tmp_path), model_folder, tmp_path)
    assert result["transcription"]["scope"]["timestamp_reference"] == "decoded_audio"
    assert "gaps or jumps" in result["message"]
    assert "may not align with video playback" in result["message"]
    assert result["segments"][0]["start"] == 0.1


def test_accumulated_small_timestamp_gaps_are_disclosed(tmp_path, model_folder, recognizer, monkeypatch):
    fake_container(monkeypatch, positions=[0, 1.04, 2.08, 3.12])
    result = transcribe(wav_file(tmp_path), model_folder, tmp_path)
    assert result["transcription"]["scope"]["timestamp_reference"] == "decoded_audio"
    assert "gaps or jumps" in result["message"]


def test_missing_timestamps_remain_explicitly_audio_relative(tmp_path, model_folder, recognizer, monkeypatch):
    fake_container(monkeypatch, positions=[None, 1])
    result = transcribe(wav_file(tmp_path), model_folder, tmp_path)
    assert result["transcription"]["scope"]["timestamp_reference"] == "decoded_audio"
    assert "Complete source timing is unavailable" in result["message"]


def test_video_without_audio_does_not_require_a_model(tmp_path, recognizer, monkeypatch):
    fake_container(monkeypatch, positions=[], no_audio=True)
    result = transcribe(wav_file(tmp_path), None, tmp_path)
    assert result["transcription"]["status"] == "no_audio"
    assert result["sha256"] and result["artifacts"] == []
    assert recognizer["models"] == []


def test_preview_is_bounded_and_full_text_remains_in_artifact(tmp_path, model_folder, recognizer):
    recognizer["result"] = {"text": "wide transcript " * 4000, "result": []}
    result = transcribe(wav_file(tmp_path), model_folder, tmp_path)
    assert result["preview_truncated"] and len(result["text"]) == 12000
    assert len(json.dumps(result).encode()) < 150000
    document = json.loads(Path(result["artifacts"][2]["path"]).read_text())
    assert len(document["text"]) > len(result["text"])
    assert "chat shows an excerpt" in result["message"]
    assert "Captions include only text with valid word timestamps" in result["message"]


def test_invalid_word_timing_is_not_exported_as_real_timestamps():
    result, clipped = speech._recognition_result(json.dumps({"text": "hello", "result": [
        {"word": "hello", "start": float("nan"), "end": 1},
        {"word": "hello", "start": -1, "end": 1},
        {"word": "hello", "start": 0.1, "end": float("inf")},
    ]}), 1)
    assert not clipped and result == {"text": "hello", "words": []}


def test_srt_hours_and_millisecond_carry():
    assert speech._srt_time(3599.9999) == "01:00:00,000"
    assert speech._srt_time(0.0006) == "00:00:00,001"
