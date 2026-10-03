"""Audio integrations: offline speech recognition via local models."""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

from sweep_core.integrations import _module_available

DEFAULT_MODEL_DIR = Path("models/vosk")


def availability() -> dict[str, Any]:
    return {
        "vosk": {"available": _module_available("vosk")},
        "model_installed": _default_model_present(),
        "whisper": {"available": _module_available("whisper")},
        "faster_whisper": {"available": _module_available("faster_whisper")},
        "ctranslate2": {"available": _module_available("ctranslate2")},
        "deepspeech": {
            "available": False,
            "reason": "archived upstream (mozilla/DeepSpeech); Vosk is the supported path",
        },
        "ultravox": {
            "available": False,
            "reason": "GPU-scale multimodal model; integrate via hosted API endpoint",
        },
    }


def _default_model_present() -> bool:
    if not DEFAULT_MODEL_DIR.exists():
        return False
    return any(DEFAULT_MODEL_DIR.glob("vosk-model*"))


def transcribe_wav(path: str | Path, model_dir: str | Path | None = None) -> dict[str, Any]:
    """Transcribe a 16kHz mono WAV file with a local Vosk model."""
    from vosk import Model, KaldiRecognizer

    model_path = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    if not (model_path / "am").exists():
        candidates = list(model_path.glob("vosk-model*"))
        if candidates and candidates[0].is_dir():
            model_path = candidates[0]
    if not model_path.exists():
        raise FileNotFoundError(
            f"no vosk model at {model_path}; run scripts/fetch_integration_assets.py"
        )
    model = Model(str(model_path))
    with wave.open(str(path), "rb") as wf:
        if wf.getframerate() != 16000 or wf.getnchannels() != 1:
            raise ValueError("input must be 16 kHz mono WAV")
        recognizer = KaldiRecognizer(model, 16000)
        recognizer.SetWords(False)
        results: list[dict[str, Any]] = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                results.append(json.loads(recognizer.Result()))
        results.append(json.loads(recognizer.FinalResult()))
    text = " ".join(part.get("text", "") for part in results).strip()
    return {"engine": "vosk", "file": str(path), "text": text}


def transcribe_with_quick_model(
    path: str | Path, *, model_size: str = "tiny", language: str | None = None
) -> dict[str, Any]:
    """Transcribe audio with a lightweight offline model (downloads on first use)."""
    import whisper

    model = whisper.load_model(model_size)
    opts: dict[str, Any] = {"fp16": False}
    if language:
        opts["language"] = language
    result = model.transcribe(str(path), **opts)
    return {
        "engine": "whisper",
        "model": model_size,
        "file": str(path),
        "text": result.get("text", "").strip(),
        "segments": len(result.get("segments", [])),
    }


def transcribe_with_accelerated_model(
    path: str | Path, *, model_size: str = "tiny", language: str | None = None
) -> dict[str, Any]:
    """Transcribe with an accelerated CTranslate2 backend model."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(path), language=language)
    text = " ".join(seg.text.strip() for seg in segments)
    return {
        "engine": "faster-whisper",
        "model": model_size,
        "file": str(path),
        "text": text.strip(),
        "language": info.language,
    }
