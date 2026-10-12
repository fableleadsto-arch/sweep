"""Bounded local media inspection without network inputs or secondary files.

Video previews are samples, not exhaustive analysis or object tracking. Timestamps
come from decoded presentation timestamps; absent timestamps stay absent.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".ogg", ".oga", ".m4a", ".aac", ".opus", ".wma"})
VIDEO_SUFFIXES = frozenset({".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".wmv", ".mpeg", ".mpg"})
MAX_BYTES = 200_000_000
MAX_ARTIFACT_BYTES = 20_000_000
MAX_DURATION_SECONDS = 1800
MAX_SECONDS = 45
MAX_PIXELS = 3840 * 2160
MAX_DECODED_PIXELS = 1_000_000_000
MAX_DECODED_FRAMES = 960
MAX_FRAMES_PER_SEEK = 180
MAX_KEYFRAMES = 8
_FORMATS = "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi,wav,mp3,flac,ogg,aac,asf,mpeg,mpegvideo"


def _local_path(value: str | Path) -> Path:
    supplied = os.fspath(value)
    if supplied.startswith(("\\\\", "//")) or "://" in supplied or "\x00" in supplied:
        raise ValueError("Choose a file stored on this computer, not a network share or URL.")
    path = Path(supplied).expanduser().absolute()
    # Check from the filesystem root downward before resolving, so a junction
    # cannot make later checks access a remote share. Alternate streams/devices
    # are not ordinary attachments on Windows.
    reserved = getattr(os.path, "isreserved", lambda value: Path(value).is_reserved())
    if os.name == "nt" and (":" in str(path)[len(path.drive):] or reserved(path)):
        raise ValueError("Attach a regular local file, not a device or alternate stream.")
    for item in [*reversed(path.parents), path]:
        if item.is_symlink() or item.is_junction():
            raise ValueError("Choose a local file or folder directly, not a symbolic link or junction.")
    return path


def read_media(path: str, granted_paths: list[str]) -> tuple[Path, bytes, dict]:
    """Read an explicitly attached local file once; never change its contents."""
    supplied = _local_path(path)
    selected = supplied.resolve(strict=True)
    allowed = {_local_path(item).resolve() for item in granted_paths}
    if selected not in allowed:
        raise PermissionError("Attach this file before asking Sweep to read it.")
    if selected.suffix.casefold() not in AUDIO_SUFFIXES | VIDEO_SUFFIXES:
        raise ValueError("Attach a supported audio or video file for media inspection.")
    if not selected.is_file():
        raise ValueError("Attach a regular local audio or video file.")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(selected, flags), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Attach a regular local audio or video file.")
        if before.st_size > MAX_BYTES:
            raise ValueError("Local media inspection supports files up to 200 MB.")
        raw = stream.read(MAX_BYTES + 1)
        after = os.fstat(stream.fileno())
    if len(raw) > MAX_BYTES:
        raise ValueError("Local media inspection supports files up to 200 MB.")
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or len(raw) != before.st_size:
        raise ValueError("The media file changed while being read. Finish saving it and attach it again.")
    return selected, raw, {
        "title": selected.name, "path": str(selected), "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "modified": datetime.fromtimestamp(before.st_mtime, UTC).isoformat(),
        "execution": "local", "source_unchanged": True,
    }


def _recognized_header(raw: bytes) -> bool:
    if len(raw) < 12:
        return False
    return (
        (raw[:4] in (b"RIFF", b"RF64") and raw[8:12] in (b"WAVE", b"AVI "))
        or raw[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free")
        or raw.startswith((b"\x1aE\xdf\xa3", b"fLaC", b"OggS", b"ID3", b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3"))
        or raw.startswith(bytes.fromhex("3026b2758e66cf11a6d900aa0062ce6c"))
        or (raw[0] == 0xFF and raw[1] & 0xE0 == 0xE0)
    )


def _deny_secondary(*_args, **_kwargs):
    raise ValueError("Media that refers to other files or network resources is not supported.")


def open_media(raw: bytes):
    """Open only in-memory media; caller must close/use the returned context."""
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_BYTES:
        raise ValueError("Provide a non-empty media file of at most 200 MB.")
    if not _recognized_header(raw):
        raise ValueError("This is not a supported standalone media file. Playlists and network streams are not accepted.")
    try:
        import av
    except ImportError as exc:
        raise ValueError("Media reading is unavailable in this installation. Install the current Sweep desktop build.") from exc
    try:
        return av.open(
            io.BytesIO(raw), mode="r", io_open=_deny_secondary,
            timeout=(10.0, 5.0),
            options={"protocol_whitelist": "sweep-memory-only", "format_whitelist": _FORMATS,
                     "probesize": "5000000", "analyzeduration": "5000000", "max_streams": "32",
                     "max_pixels": str(MAX_PIXELS), "threads": "2", "fflags": "-genpts"},
        )
    except Exception as exc:
        raise ValueError("This media file could not be opened. It may be damaged, unsupported, or refer to other resources.") from exc


def write_artifact(artifact_dir: str | Path, name_stem: str, suffix: str, payload: bytes) -> dict:
    """Write a bounded new artifact using an exclusive name, never an overwrite."""
    if not isinstance(payload, bytes) or len(payload) > MAX_ARTIFACT_BYTES:
        raise ValueError("The generated media artifact exceeds the 20 MB output limit.")
    if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
        raise ValueError("The media artifact has an unsupported file extension.")
    target_dir = _local_path(artifact_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_dir = _local_path(target_dir).resolve(strict=True)
    safe_stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", name_stem).strip("-_")[:60] or "media"
    target = target_dir / f"{safe_stem}-{uuid.uuid4().hex}{suffix}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(target, flags, 0o600), "wb") as stream:
        stream.write(payload)
    return {"path": str(target), "name": safe_stem + suffix, "format": suffix.lstrip("."),
            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _seconds(value, time_base):
    return _number(value * time_base) if value is not None and time_base is not None else None


def _timestamp(frame):
    return _seconds(frame.pts, frame.time_base)


def _label(timestamp):
    if timestamp is None:
        return "Timestamp unavailable"
    sign = "-" if timestamp < 0 else ""
    milliseconds = round(abs(timestamp) * 1000)
    seconds, ms = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{sign}{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


def _stream_details(stream) -> dict:
    codec = stream.codec_context
    details = {"index": stream.index, "type": stream.type,
               "codec": str(codec.name or "unknown")[:80],
               "duration_seconds": _seconds(stream.duration, stream.time_base),
               "start_seconds": _seconds(stream.start_time, stream.time_base)}
    if stream.type == "video":
        details.update(width=codec.width, height=codec.height, frame_rate=_number(stream.average_rate))
    elif stream.type == "audio":
        details.update(sample_rate=codec.sample_rate, channels=len(codec.layout.channels))
    return details


def _metadata_summary(result: dict) -> str:
    duration = result["duration_seconds"]
    length = f"{duration:.2f} seconds" if duration is not None else "unknown"
    lines = [f"Duration: {length} · Format: {result['format']}"]
    for stream in result["streams"][:8]:
        if stream["type"] == "video":
            rate = stream.get("frame_rate")
            speed = f" · {rate:g} fps" if rate is not None else ""
            lines.append(f"Video: {stream['width']} × {stream['height']}{speed} · {stream['codec']}")
        else:
            lines.append(f"Audio: {stream['sample_rate']} Hz · {stream['channels']} channel(s) · {stream['codec']}")
    return "\n".join(lines)


class _BudgetExceeded(ValueError):
    pass


def _check_time(start):
    if time.monotonic() - start > MAX_SECONDS:
        raise _BudgetExceeded("The local preview time limit was reached.")


def _sample_frames(container, stream, duration, start):
    codec = stream.codec_context
    if codec.width <= 0 or codec.height <= 0 or codec.width * codec.height > MAX_PIXELS:
        raise ValueError("Video previews support dimensions up to 8.3 million pixels. Attach a smaller copy.")
    codec.thread_count = 2
    codec.options = {**codec.options, "max_pixels": str(MAX_PIXELS)}
    first = _seconds(stream.start_time, stream.time_base)
    origin = first if first is not None else 0.0
    targets = [origin + duration * index / MAX_KEYFRAMES for index in range(MAX_KEYFRAMES)] if duration else [None]
    decoded = pixels = 0
    samples = []
    warnings = []
    seen = set()
    try:
        for target in targets:
            _check_time(start)
            seek_frames = 0
            if target is not None and stream.time_base:
                try:
                    container.seek(int(target / stream.time_base), stream=stream, backward=True, any_frame=False)
                except Exception:
                    warnings.append("Some sample positions could not be reached in this file.")
                    break
            reached = False
            for frame in container.decode(stream):
                _check_time(start)
                decoded += 1
                seek_frames += 1
                area = frame.width * frame.height
                pixels += area
                if area <= 0 or area > MAX_PIXELS:
                    raise ValueError("A decoded frame exceeds the video dimension limit.")
                if decoded > MAX_DECODED_FRAMES or pixels > MAX_DECODED_PIXELS:
                    raise _BudgetExceeded("The local preview decoding limit was reached.")
                timestamp = _timestamp(frame)
                if target is not None and timestamp is not None and timestamp < target:
                    if seek_frames >= MAX_FRAMES_PER_SEEK:
                        raise _BudgetExceeded("A sample position requires too much decoding; attach a shorter clip.")
                    continue
                identity = timestamp if timestamp is not None else ("unknown", decoded)
                if identity not in seen:
                    scale = min(960 / frame.width, 960 / frame.height, 1.0)
                    image = frame.to_image(width=max(1, round(frame.width * scale)), height=max(1, round(frame.height * scale)))
                    rotation = frame.rotation
                    if rotation:
                        image = image.rotate(rotation, expand=True)
                    image.thumbnail((960, 540))
                    seen.add(identity)
                    samples.append((timestamp, image, target))
                reached = True
                if target is not None or len(samples) >= MAX_KEYFRAMES:
                    break
            if target is not None and not reached:
                warnings.append("Some sample positions had no decodable frame.")
            if len(samples) >= MAX_KEYFRAMES:
                break
    except _BudgetExceeded as exc:
        warnings.append(str(exc))
    if not samples:
        raise ValueError("No video preview could be decoded within the local limits. Attach a shorter or smaller clip.")
    return samples, warnings


def inspect_media(path: str, granted_paths: list[str], artifact_dir: str | Path, *, progress=None) -> dict:
    """Inspect local audio metadata or export up to eight timestamped video samples."""
    selected, raw, result = read_media(path, granted_paths)
    start = time.monotonic()
    if progress:
        progress("Reading local media details")
    try:
        with open_media(raw) as container:
            _check_time(start)
            streams = [stream for stream in container.streams if stream.type in {"audio", "video"}]
            if not streams:
                raise ValueError("No supported audio or video stream was found in this file.")
            duration = _seconds(container.duration, 0.000001)
            if duration is None:
                durations = [_seconds(stream.duration, stream.time_base) for stream in streams]
                duration = max((value for value in durations if value is not None and value >= 0), default=None)
            if duration is not None and duration > MAX_DURATION_SECONDS:
                raise ValueError("Local media inspection supports clips up to 30 minutes. Attach a shorter extract.")
            if duration is not None and duration < 0:
                duration = None
            result.update(format=str(container.format.name)[:120], duration_seconds=duration,
                          streams=[_stream_details(stream) for stream in streams[:32]],
                          metadata={key: str(value)[:512] for key, value in container.metadata.items()
                                    if key.casefold() in {"title", "artist", "album", "date", "creation_time", "comment"}},
                          keyframes=[], artifacts=[], warnings=[])
            video = next((stream for stream in streams if stream.type == "video" and not (stream.disposition.value & 0x0400)), None)
            if video is None:
                result["message"] = f"Read {selected.name} locally. Audio metadata inspected; no transcription was performed."
                result["message"] += "\n\n" + _metadata_summary(result)
                return result
            if progress:
                progress("Sampling video frames locally")
            video_duration = _seconds(video.duration, video.time_base)
            if video_duration is None or video_duration <= 0:
                video_duration = duration
            samples, warnings = _sample_frames(container, video, video_duration, start)
            result["warnings"].extend(warnings)
    except (ValueError, PermissionError):
        raise
    except Exception as exc:
        raise ValueError("The media could not be decoded. It may be damaged or use an unsupported codec.") from exc

    from PIL import Image, ImageDraw

    tile_width, tile_height = 480, 294
    columns = min(2, len(samples))
    sheet = Image.new("RGB", (columns * tile_width, math.ceil(len(samples) / columns) * tile_height), "#111827")
    draw = ImageDraw.Draw(sheet)
    manifest_frames = []
    # Sampling may stop at its time limit with useful frames already decoded.
    # Export those bounded samples (at most eight 960x540 images) even when the
    # sampling deadline has elapsed; the artifact byte limit still applies.
    for index, (timestamp, image, target) in enumerate(samples):
        payload = io.BytesIO()
        image.save(payload, "JPEG", quality=85)
        artifact = write_artifact(artifact_dir, f"{selected.stem}-frame-{index + 1}", ".jpg", payload.getvalue())
        result["artifacts"].append(artifact)
        label = _label(timestamp)
        result["keyframes"].append({"timestamp": timestamp, "label": label, "path": artifact["path"], "sha256": artifact["sha256"]})
        manifest_frames.append({"timestamp": timestamp, "label": label, "requested_timestamp": target,
                                "file": Path(artifact["path"]).name, "sha256": artifact["sha256"]})
        thumbnail = image.copy()
        thumbnail.thumbnail((480, 270))
        left = index % columns * tile_width
        top = index // columns * tile_height
        sheet.paste(thumbnail, (left + (tile_width - thumbnail.width) // 2, top + (270 - thumbnail.height) // 2))
        draw.text((left + 10, top + 275), label, fill="white")
    sheet_bytes = io.BytesIO()
    sheet.save(sheet_bytes, "PNG")
    contact_sheet = write_artifact(artifact_dir, f"{selected.stem}-contact-sheet", ".png", sheet_bytes.getvalue())
    result["artifacts"].insert(0, contact_sheet)
    result["contact_sheet"] = contact_sheet
    limitation = "Sampled frames are not exhaustive analysis, object tracking, or proof of events between samples."
    if duration is None:
        result["warnings"].append("Duration is unavailable; only the first decodable frames were sampled.")
    if any(timestamp is None for timestamp, _image, _target in samples):
        result["warnings"].append("Some frames have no presentation timestamp; their times are left unknown.")
    result["sampling"] = {"method": "distributed seeks" if duration else "initial frames", "frames": len(samples),
                          "timestamps": "decoded presentation timestamps, in seconds on the original media timeline",
                          "exhaustive": False, "object_tracking": False, "limitations": limitation}
    manifest = {"source_sha256": result["sha256"], "source_name": selected.name,
                "duration_seconds": duration, "sampling": result["sampling"],
                "warnings": result["warnings"], "frames": manifest_frames}
    result["artifacts"].append(write_artifact(artifact_dir, f"{selected.stem}-frames", ".json",
                                             json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")))
    result["message"] = f"Inspected {selected.name} locally and saved {len(samples)} sampled video frames and a contact sheet. {limitation}"
    result["message"] += "\n\n" + _metadata_summary(result)
    if result["warnings"]:
        result["message"] += " " + " ".join(result["warnings"])
    return result
