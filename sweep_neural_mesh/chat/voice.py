"""Voice I/O for Sweep chat.

Input:  microphone (pyaudio) -> WAV frames -> Whisper (tiny.en, CPU) -> text.
        The existing neurons.speech_recognition.SpeechRecognizer does the STT;
        this module adds mic capture, energy-based VAD (silence detection),
        and push-to-talk / continuous modes.
Output: pyttsx3 text-to-speech (SAPI5 voices on Windows), off-thread so the
        chat loop never blocks while speaking.

No mic on the machine? Every entry point degrades gracefully with a clear
error the CLI can show.
"""
from __future__ import annotations

import io
import logging
import struct
import threading
import time
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

# 16 kHz mono 16-bit — Whisper's native format
RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2
FRAME_MS = 30                      # 30 ms frames for VAD
FRAME_SAMPLES = RATE * FRAME_MS // 1000
SILENCE_THRESHOLD = 450            # RMS energy below this = silence
MAX_RECORD_SECONDS = 20
SILENCE_HANGOVER_MS = 900          # stop after ~0.9 s of trailing silence


def _check_mic() -> None:
    try:
        import pyaudio  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Microphone capture unavailable (pyaudio missing). "
            "Use text chat, or install: pip install pyaudio"
        ) from e


def list_microphones() -> list[str]:
    """Enumerate input devices (for debugging / picking a device index)."""
    _check_mic()
    import pyaudio
    pa = pyaudio.PyAudio()
    try:
        return [
            f"[{i}] {pa.get_device_info_by_index(i).get('name', '?')}"
            for i in range(pa.get_device_count())
            if pa.get_device_info_by_index(i).get("maxInputChannels", 0) > 0
        ]
    finally:
        pa.terminate()


def record_utterance(device_index: int | None = None,
                     continuous: bool = False) -> bytes:
    """Record one utterance from the microphone.

    Push-to-talk mode (continuous=False): records until MAX_RECORD_SECONDS
    or ~1 s of trailing silence.
    Continuous mode: waits for speech to start (energy gate), then records
    until the trailing silence.

    Returns raw 16 kHz mono 16-bit PCM bytes. Raises RuntimeError if the
    mic can't be opened.
    """
    _check_mic()
    import pyaudio

    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=pyaudio.paInt16, channels=CHANNELS, rate=RATE,
            input=True, input_device_index=device_index,
            frames_per_buffer=FRAME_SAMPLES,
        )
    except Exception as e:
        pa.terminate()
        raise RuntimeError(f"Could not open microphone: {e}") from e

    frames: list[bytes] = []
    silent_ms = 0
    started = continuous is False  # push-to-talk starts immediately
    speech_ms = 0
    t0 = time.time()
    try:
        while True:
            data = stream.read(FRAME_SAMPLES, exception_on_overflow=False)
            frames.append(data)
            rms = _rms(data)
            if rms > SILENCE_THRESHOLD:
                speech_ms += FRAME_MS
                silent_ms = 0
                started = True
            else:
                silent_ms += FRAME_MS
            elapsed = time.time() - t0
            if elapsed > MAX_RECORD_SECONDS:
                break
            if started and speech_ms > 250 and silent_ms >= SILENCE_HANGOVER_MS:
                break
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    return b"".join(frames)


def _rms(pcm: bytes) -> float:
    n = len(pcm) // SAMPLE_WIDTH
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", pcm[: n * SAMPLE_WIDTH])
    return (sum(s * s for s in samples) / n) ** 0.5


def pcm_to_wav_bytes(pcm: bytes) -> io.BytesIO:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(RATE)
        w.writeframes(pcm)
    buf.seek(0)
    return buf


def transcribe(pcm: bytes):
    """PCM bytes -> TranscriptResult via the existing Whisper wrapper."""
    from sweep_neural_mesh.neurons.speech_recognition import SpeechRecognizer
    rec = SpeechRecognizer()
    wav = pcm_to_wav_bytes(pcm)
    return rec.recognize(wav)


# ── TTS ──────────────────────────────────────────────────────────────

class Speaker:
    """pyttsx3 wrapper; speak() is fire-and-forget, stop() cancels."""

    def __init__(self, rate: int = 175, voice_substring: str | None = None):
        self._rate = rate
        self._voice_substring = voice_substring
        self._engine = None
        self._lock = threading.Lock()
        self._speaking = False
        self.enabled = True

    def _get_engine(self):
        if self._engine is None:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self._rate)
            if self._voice_substring:
                for v in self._engine.getProperty("voices"):
                    if self._voice_substring.lower() in v.name.lower():
                        self._engine.setProperty("voice", v.id)
                        break
        return self._engine

    def speak(self, text: str) -> None:
        """Speak on a daemon thread; long text is truncated sensibly."""
        if not self.enabled or not text:
            return
        clean = _strip_markdown(text)[:400]
        if not clean:
            return

        def worker():
            with self._lock:
                self._speaking = True
                try:
                    eng = self._get_engine()
                    eng.say(clean)
                    eng.runAndWait()
                except Exception as e:
                    logger.warning("TTS failed: %s", e)
                finally:
                    self._speaking = False

        threading.Thread(target=worker, daemon=True).start()

    @property
    def speaking(self) -> bool:
        return self._speaking

    def stop(self) -> None:
        try:
            if self._engine is not None:
                self._engine.stop()
        except Exception:
            pass


def _strip_markdown(text: str) -> str:
    import re
    text = re.sub(r"```[\s\S]*?```", " code block omitted. ", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\[(.*?)\]\([^)]*\)", r"\1", text)
    return text.strip()
