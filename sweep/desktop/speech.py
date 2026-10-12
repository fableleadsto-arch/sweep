"""Bounded offline transcription of explicitly attached local media.

Recognition loads only the model folder installed through local speech setup.
No language-based model lookup, download, cloud client, microphone capture or
speaker identification is performed here. Timestamps are derived from recognized
words and decoded audio.
"""

from __future__ import annotations

import json
import math
import os
import stat
import time
from pathlib import Path

from .media import AUDIO_SUFFIXES, VIDEO_SUFFIXES, open_media, read_media, write_artifact

SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 300
MAX_PROCESS_SECONDS = 240
MAX_DECODE_FRAMES = 100_000
MAX_WORDS = 10_000
MAX_TEXT = 80_000
MAX_SEGMENTS = 1000
MAX_MODEL_BYTES = 1_500_000_000
MAX_MODEL_FILES = 4096
MODEL_HELP = "Open Settings and choose Set up speech, then try this file again."
LIMITATIONS = (
    "Automatic transcription can mishear words, names and numbers. Review it against the audio. "
    "Only the first audio track is used; speakers are not identified."
)


def _local_path(value: str | Path) -> Path:
    text = str(value)
    if not text or text.startswith(("\\\\", "//")):
        raise ValueError(MODEL_HELP)
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise ValueError(MODEL_HELP)
    if "://" in text or "\x00" in text or (os.name == "nt" and ":" in str(path)[len(path.drive):]):
        raise ValueError(MODEL_HELP)
    for item in [*reversed(path.parents), path]:
        if item.is_symlink() or item.is_junction():
            raise ValueError("Choose a speech model stored directly on this computer, not a link. " + MODEL_HELP)
    return path


def validate_model_path(model_path: str | Path | None) -> Path:
    """Validate a bounded, self-contained Vosk folder without loading the engine."""
    if not model_path:
        raise ValueError(MODEL_HELP)
    try:
        path = _local_path(model_path)
        if path.is_symlink() or path.is_junction() or not path.is_dir():
            raise ValueError(MODEL_HELP)
        resolved = path.resolve(strict=True)
        if str(resolved).startswith(("\\\\", "//")):
            raise ValueError(MODEL_HELP)
        pending = [(resolved, 0)]
        count = total = 0
        while pending:
            directory, depth = pending.pop()
            if depth > 12:
                raise ValueError("The speech model folder contains too many nested folders. " + MODEL_HELP)
            with os.scandir(directory) as entries:
                for entry in entries:
                    count += 1
                    if count > MAX_MODEL_FILES:
                        raise ValueError("The speech model folder contains too many files. " + MODEL_HELP)
                    info = entry.stat(follow_symlinks=False)
                    if entry.is_symlink() or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
                        raise ValueError("The speech model must contain local files, not links. " + MODEL_HELP)
                    if stat.S_ISDIR(info.st_mode):
                        pending.append((Path(entry.path), depth + 1))
                    elif stat.S_ISREG(info.st_mode):
                        total += info.st_size
                        if total > MAX_MODEL_BYTES:
                            raise ValueError("The speech model exceeds the 1.5 GB local model limit. " + MODEL_HELP)
                    else:
                        raise ValueError("The speech model must contain regular local files. " + MODEL_HELP)
        # Verify filenames only after rejecting nested links and junctions; an
        # eager is_file() on a required path could otherwise follow a share.
        required = ["am/final.mdl", "conf/model.conf", "conf/mfcc.conf", "graph/phones/word_boundary.int"]
        if (resolved / "graph/HCLG.fst").is_file():
            required.append("graph/HCLG.fst")
        else:
            required.extend(("graph/HCLr.fst", "graph/Gr.fst"))
        if any(not (resolved / name).is_file() or (resolved / name).stat().st_size == 0 for name in required):
            raise ValueError(MODEL_HELP)
        return resolved
    except OSError as exc:
        raise ValueError(MODEL_HELP) from exc


def _finite(value) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _timestamp(frame) -> float | None:
    if frame.pts is None or frame.time_base is None:
        return None
    return _finite(frame.pts * frame.time_base)


def _srt_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def _recognition_result(raw: str, processed_seconds: float) -> tuple[dict | None, bool]:
    if not isinstance(raw, str) or len(raw) > 2_000_000:
        raise ValueError("Local speech recognition exceeded its response limit.")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Local speech recognition returned an invalid transcript.")
    if not isinstance(data.get("text", ""), str):
        raise ValueError("Local speech recognition returned an invalid transcript.")
    text = " ".join(data.get("text", "").split())
    if not text:
        return None, False
    clipped = len(text) > MAX_TEXT
    text = text[:MAX_TEXT]
    words = []
    values = data.get("result", [])
    if not isinstance(values, list):
        values = []
    for word in values[:MAX_WORDS]:
        if not isinstance(word, dict):
            continue
        start, end = _finite(word.get("start")), _finite(word.get("end"))
        token = str(word.get("word", ""))[:500].strip()
        if start is None or end is None or not token or start < 0 or end <= start or start >= processed_seconds:
            continue
        item = {"start": round(start, 3), "end": round(min(end, processed_seconds), 3), "word": token}
        confidence = _finite(word.get("conf"))
        if confidence is not None and 0 <= confidence <= 1:
            item["confidence"] = round(confidence, 4)
        words.append(item)
    clipped = clipped or len(values) > MAX_WORDS
    item = {"text": text, "words": words}
    if words:
        item.update(start=words[0]["start"], end=words[-1]["end"])
    return item, clipped


def _scope_message(scope: dict) -> str:
    duration = scope["processed_seconds"]
    if scope["timestamp_reference"] == "media_playback":
        location = f"media time {scope['start_seconds']:.2f}–{scope['end_seconds']:.2f} seconds"
    else:
        location = f"{duration:.2f} seconds from the start of the decoded audio"
    return f"Processed {location}. " + scope["timestamp_note"]


def transcribe_media(path: str, grants: list[str], artifact_dir: str | Path,
                     model_path: str | Path | None, *, progress=None) -> dict:
    """Transcribe at most five minutes from an attached audio or video file."""
    selected, raw, result = read_media(path, grants)
    if selected.suffix.lower() not in AUDIO_SUFFIXES | VIDEO_SUFFIXES:
        raise ValueError("Attach a supported audio or video file to transcribe it.")
    result.update(kind="transcript", text="", segments=[], artifacts=[])
    started = time.monotonic()
    try:
        import av
        import vosk
    except ImportError as exc:
        raise ValueError("Local speech tools are missing. Install the current Sweep desktop build.") from exc
    with open_media(raw) as container:
        streams = list(container.streams.audio)
        if not streams:
            result["transcription"] = {"status": "no_audio", "message": "This file has no audio track to transcribe."}
            result["message"] = result["transcription"]["message"]
            return result
        stream = streams[0]
        result["audio"] = {"track": 1, "tracks_available": len(streams), "sample_rate": stream.codec_context.sample_rate,
                           "channels": stream.codec_context.channels, "codec": stream.codec_context.name}
        try:
            folder = validate_model_path(model_path)
        except ValueError as exc:
            result["transcription"] = {"status": "unavailable", "message": str(exc)}
            result["message"] = str(exc)
            return result
        if progress:
            progress("Loading speech recognition on this computer")
        try:
            # Passing model_path explicitly prevents Vosk's language/name based
            # downloader from ever being selected, including on a missing model.
            vosk.SetLogLevel(-1)
            model = vosk.Model(model_path=str(folder))
            recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE)
            recognizer.SetWords(True)
        except Exception as exc:
            raise ValueError("The selected speech model could not be loaded. " + MODEL_HELP) from exc
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        processed = 0
        decoded = 0
        timed_frames = 0
        source_seconds = 0.0
        first_pts = None
        expected_pts = None
        timing_continuous = True
        timing_present = True
        segments = []
        word_count = text_count = 0
        reasons = []
        last_progress = 0.0

        def collect(raw_result: str) -> None:
            nonlocal word_count, text_count
            item, clipped = _recognition_result(raw_result, processed / SAMPLE_RATE)
            if item is None:
                return
            if len(segments) >= MAX_SEGMENTS or text_count + len(item["text"]) > MAX_TEXT or word_count + len(item["words"]) > MAX_WORDS:
                reasons.append("The transcript reached its output limit.")
                return
            segments.append(item)
            word_count += len(item["words"])
            text_count += len(item["text"])
            if clipped:
                reasons.append("The transcript reached its output limit.")

        def consume(frame) -> bool:
            nonlocal processed, last_progress
            remaining = MAX_AUDIO_SECONDS * SAMPLE_RATE - processed
            samples = min(frame.samples, remaining)
            if samples <= 0:
                reasons.append("Only the first five minutes of decoded audio were transcribed.")
                return False
            payload = bytes(frame.planes[0])[:samples * 2]
            # Small batches bound recognizer calls and provide useful task progress.
            for offset in range(0, len(payload), 8000):
                chunk = payload[offset:offset + 8000]
                processed += len(chunk) // 2
                if recognizer.AcceptWaveform(chunk):
                    collect(recognizer.Result())
                if time.monotonic() - started > MAX_PROCESS_SECONDS:
                    reasons.append("Speech processing reached its four-minute time limit.")
                    return False
                if reasons:
                    return False
            elapsed = processed / SAMPLE_RATE
            if progress and elapsed - last_progress >= 10:
                progress(f"Transcribing audio locally · {elapsed:.0f} seconds processed")
                last_progress = elapsed
            if samples < frame.samples:
                reasons.append("Only the first five minutes of decoded audio were transcribed.")
                return False
            return True

        def media_failure(exc: Exception, operation: str) -> None:
            if processed == 0:
                raise ValueError(f"Audio {operation} could not begin. The file may be damaged or use an unsupported format.") from exc
            reasons.append(f"Audio {operation} stopped at an unreadable or unsupported section. "
                           "The partial transcript covers only audio processed before the error.")

        def convert(frame):
            try:
                return resampler.resample(frame)
            except (av.error.FFmpegError, ValueError) as exc:
                media_failure(exc, "conversion")
                return []

        decoder = iter(container.decode(stream))
        while True:
            # Limit recovery to the media-library boundary. Recognition errors,
            # application bugs and cancellation must still propagate normally.
            try:
                frame = next(decoder)
            except StopIteration:
                break
            except (av.error.FFmpegError, ValueError) as exc:
                media_failure(exc, "decoding")
                break
            decoded += 1
            if decoded > MAX_DECODE_FRAMES or time.monotonic() - started > MAX_PROCESS_SECONDS:
                reasons.append("Speech decoding reached its processing limit.")
                break
            if frame.samples <= 0:
                continue
            # Bound a single decoded frame before asking the resampler to allocate.
            if not 1000 <= frame.sample_rate <= 384_000 or frame.samples > frame.sample_rate * 10:
                reasons.append("The audio contains an unsupported or oversized decoded frame.")
                break
            stamp = _timestamp(frame)
            if timed_frames == 0:
                first_pts = stamp
            timed_frames += 1
            if stamp is None:
                timing_present = False
            elif expected_pts is not None and abs(stamp - expected_pts) > 0.05:
                timing_continuous = False
            if stamp is not None and first_pts is not None and abs(stamp - first_pts - source_seconds) > 0.05:
                timing_continuous = False
            expected_pts = stamp + frame.samples / frame.sample_rate if stamp is not None else None
            source_seconds += frame.samples / frame.sample_rate
            # Decode output is concatenated for recognition. Inspect source PTS
            # above, then clear it so discontinuities do not crash the resampler.
            frame.pts = None
            keep_going = True
            for converted in convert(frame):
                if not consume(converted):
                    keep_going = False
                    break
            if not keep_going or reasons:
                break
        if not reasons:
            for converted in convert(None):
                if not consume(converted):
                    break
        collect(recognizer.FinalResult())
        container_start = _finite(container.start_time)
        media_start = first_pts - container_start / av.time_base if first_pts is not None and container_start is not None else None
        known_timing = timing_present and timing_continuous and media_start is not None and media_start >= -0.05
        start_seconds = max(0.0, media_start) if known_timing else 0.0

    if known_timing:
        note = "Timestamps are relative to the start of media playback."
    elif not timing_continuous:
        note = "Source timestamps contain gaps or jumps. Transcript timestamps follow concatenated decoded audio and may not align with video playback."
    else:
        note = "Complete source timing is unavailable. Transcript timestamps are relative to decoded audio and may not align with video playback."
    for segment in segments:
        if "start" in segment:
            segment["start"] = round(segment["start"] + start_seconds, 3)
            segment["end"] = round(segment["end"] + start_seconds, 3)
        for word in segment["words"]:
            word["start"] = round(word["start"] + start_seconds, 3)
            word["end"] = round(word["end"] + start_seconds, 3)
    scope = {"processed_seconds": round(processed / SAMPLE_RATE, 3), "start_seconds": round(start_seconds, 3),
             "end_seconds": round(start_seconds + processed / SAMPLE_RATE, 3), "max_seconds": MAX_AUDIO_SECONDS,
             "truncated": bool(reasons), "timestamp_reference": "media_playback" if known_timing else "decoded_audio",
             "timestamp_note": note, "audio_track": 1, "warnings": list(dict.fromkeys(reasons))}
    text = "\n".join(item["text"] for item in segments)
    untimed = sum("start" not in item for item in segments)
    if untimed:
        scope["warnings"].append("Word timing was unavailable for some recognized text. Captions include only text with valid word timestamps; the text and JSON files retain all recognized text.")
    status = "partial" if reasons else ("completed" if text else "no_speech")
    transcription = {"status": status, "scope": scope, "limitations": LIMITATIONS}
    document = {"source": {key: result[key] for key in ("title", "sha256", "bytes") if key in result},
                "execution": "local", "transcription": transcription, "text": text, "segments": segments}
    srt = []
    for segment in segments:
        if "start" in segment:
            srt.append(f"{len(srt) + 1}\n{_srt_time(segment['start'])} --> {_srt_time(segment['end'])}\n{segment['text']}\n")
    artifacts = []
    transcript_header = _scope_message(scope) + "\n" + LIMITATIONS
    if scope["warnings"]:
        transcript_header += "\n" + " ".join(scope["warnings"])
    for suffix, payload in (
        (".txt", (transcript_header + "\n\n" + (text or "No speech was recognized.") + "\n").encode("utf-8")),
        (".srt", "\n".join(srt).encode("utf-8")),
        (".json", json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")),
    ):
        artifacts.append(write_artifact(artifact_dir, selected.stem + "-transcript", suffix, payload))
    # Complete word timestamps live in JSON. The dock event stays bounded even
    # for long transcripts; its displayed excerpt is explicitly labelled.
    preview = text[:12_000]
    displayed_segments = [{key: value for key, value in item.items() if key != "words"} for item in segments[:100]]
    for item in displayed_segments:
        item["text"] = item["text"][:500]
    summary = (preview or "No speech was recognized in the processed audio.")
    summary += "\n\n" + _scope_message(scope)
    if scope["warnings"]:
        summary += "\n" + " ".join(scope["warnings"])
    if len(text) > len(preview):
        summary += "\nThe chat shows an excerpt. Open the saved transcript for all recognized text."
    summary += "\n" + LIMITATIONS
    result.update(text=preview, segments=displayed_segments, transcription=transcription, artifacts=artifacts,
                  preview_truncated=len(text) > len(preview) or len(segments) > 100,
                  message=summary)
    return result
