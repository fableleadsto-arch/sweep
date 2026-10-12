"""Real bounded media fixtures, attachment rights and independent artifacts."""
import hashlib
import io
import json
import subprocess
import sys
import wave
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from sweep.desktop import media

av = pytest.importorskip("av")


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "one-second.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\0\0" * 16_000)
    return path


@pytest.fixture
def video_file(tmp_path):
    path = tmp_path / "colors.mp4"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=10)
        stream.width, stream.height = 320, 180
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = 10
        for index in range(80):
            frame = av.VideoFrame.from_image(Image.new("RGB", (320, 180), (index * 3, 40, 255 - index * 3)))
            frame.pts = index + 50
            frame.time_base = Fraction(1, 10)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return path


def inspect(path, tmp_path, **kwargs):
    return media.inspect_media(str(path), [str(path)], tmp_path / "artifacts", **kwargs)


def test_audio_metadata_is_local_and_source_is_preserved(audio_file, tmp_path):
    original = audio_file.read_bytes()
    result = inspect(audio_file, tmp_path)
    assert result["execution"] == "local"
    assert result["source_unchanged"] is True
    assert result["sha256"] == hashlib.sha256(original).hexdigest()
    assert result["duration_seconds"] == pytest.approx(1)
    assert result["streams"][0]["sample_rate"] == 16_000
    assert result["streams"][0]["channels"] == 1
    assert not result["keyframes"] and not result["artifacts"]
    assert "no transcription" in result["message"]
    assert "1.00 seconds" in result["message"] and "16000 Hz" in result["message"]
    assert audio_file.read_bytes() == original
    assert not (tmp_path / "artifacts").exists()


def test_real_video_samples_have_actual_pts_and_standalone_artifacts(video_file, tmp_path):
    original = video_file.read_bytes()
    with av.open(io.BytesIO(original)) as container:
        real_times = {float(frame.pts * frame.time_base) for frame in container.decode(video=0)}
    progress = []
    result = inspect(video_file, tmp_path, progress=progress.append)
    assert 2 <= len(result["keyframes"]) <= 8
    assert progress == ["Reading local media details", "Sampling video frames locally"]
    assert result["sampling"]["exhaustive"] is False
    assert result["sampling"]["object_tracking"] is False
    timestamps = [frame["timestamp"] for frame in result["keyframes"]]
    assert timestamps == sorted(set(timestamps))
    assert set(timestamps) <= real_times
    assert timestamps[0] >= 5  # An actual nonzero presentation origin, not frame / fps.
    assert timestamps[-1] - timestamps[0] >= 6
    assert "not exhaustive" in result["message"]
    for artifact in result["artifacts"]:
        payload = Path(artifact["path"]).read_bytes()
        assert artifact["bytes"] == len(payload)
        assert artifact["sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["contact_sheet"] == result["artifacts"][0]
    with Image.open(result["contact_sheet"]["path"]) as sheet:
        assert sheet.format == "PNG" and sheet.width <= 960
    for frame in result["keyframes"]:
        with Image.open(frame["path"]) as image:
            assert image.format == "JPEG" and image.width <= 960 and image.height <= 540
    manifest_artifact = next(item for item in result["artifacts"] if item["format"] == "json")
    manifest = json.loads(Path(manifest_artifact["path"]).read_text())
    assert manifest["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert [frame["timestamp"] for frame in manifest["frames"]] == timestamps
    for frame in manifest["frames"]:
        saved = Path(manifest_artifact["path"]).parent / frame["file"]
        assert saved.is_file() and hashlib.sha256(saved.read_bytes()).hexdigest() == frame["sha256"]
    assert video_file.read_bytes() == original


def test_numpy_is_not_needed_for_real_video_inspection(video_file, tmp_path):
    script = """
import builtins, sys
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'numpy' or name.startswith('numpy.'):
        raise ImportError('numpy deliberately unavailable in desktop')
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
from sweep.desktop.media import inspect_media
result = inspect_media(sys.argv[1], [sys.argv[1]], sys.argv[2])
assert result['keyframes']
print(len(result['keyframes']))
"""
    run = subprocess.run([sys.executable, "-c", script, str(video_file), str(tmp_path / "numpy-free")],
                         capture_output=True, text=True, timeout=45)
    assert run.returncode == 0, run.stderr


@pytest.mark.parametrize("grant", [None, "directory", "different"])
def test_exact_file_grant_is_required(audio_file, tmp_path, grant):
    grants = [] if grant is None else [str(tmp_path if grant == "directory" else tmp_path / "other.wav")]
    with pytest.raises(PermissionError, match="Attach this file"):
        media.inspect_media(str(audio_file), grants, tmp_path / "artifacts")
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("path", [r"\\server\share\a.mp4", "//server/share/a.mp4", "https://host/media.mp4", r"\\?\C:\file.mp4"])
def test_remote_and_device_paths_are_rejected_before_resolution(path, monkeypatch):
    monkeypatch.setattr(Path, "resolve", lambda *_args, **_kwargs: pytest.fail("must not resolve remote input"))
    with pytest.raises(ValueError, match="network share or URL"):
        media.read_media(path, [path])


def test_remote_grant_is_not_resolved(audio_file, monkeypatch):
    original = Path.resolve

    def resolving(path, *args, **kwargs):
        assert not str(path).startswith(("\\\\", "//"))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolving)
    with pytest.raises(ValueError, match="network share"):
        media.read_media(str(audio_file), [r"\\server\share\file.wav"])


def test_file_size_limit_is_enforced_before_decoder(audio_file, monkeypatch):
    monkeypatch.setattr(media, "MAX_BYTES", 20)
    monkeypatch.setattr(media, "open_media", lambda *_args: pytest.fail("must not open oversized file"))
    with pytest.raises(ValueError, match="200 MB"):
        media.read_media(str(audio_file), [str(audio_file)])


def test_non_regular_file_and_unknown_suffix_are_rejected(tmp_path):
    folder = tmp_path / "folder.mp4"
    folder.mkdir()
    with pytest.raises(ValueError, match="regular local"):
        media.read_media(str(folder), [str(folder)])
    text = tmp_path / "input.txt"
    text.write_text("hello")
    with pytest.raises(ValueError, match="supported audio or video"):
        media.read_media(str(text), [str(text)])


@pytest.mark.parametrize("payload", [
    b"#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1000\nhttp://127.0.0.1/secret\n",
    b"ffconcat version 1.0\nfile 'file:///private/file.mp4'\n",
    b"<MPD><BaseURL>https://host/</BaseURL></MPD>",
    b"https://host/file.mp4", b"dshow:video=Camera", b"\0" * 20,
])
def test_playlists_urls_and_devices_are_rejected_without_decoder(payload, monkeypatch):
    monkeypatch.setattr(av, "open", lambda *_args, **_kwargs: pytest.fail("decoder must not open playlist"))
    with pytest.raises(ValueError, match="standalone media"):
        media.open_media(payload)


def test_open_media_denies_secondary_resources_and_network_protocols(audio_file, monkeypatch):
    original = av.open

    def audited(buffer, **kwargs):
        assert isinstance(buffer, io.BytesIO)
        assert kwargs["options"]["protocol_whitelist"] == "sweep-memory-only"
        assert "hls" not in kwargs["options"]["format_whitelist"]
        assert "concat" not in kwargs["options"]["format_whitelist"]
        for url in ["file:///private/other.mp4", "http://127.0.0.1/", "pipe:0", "relative.wav"]:
            with pytest.raises(ValueError, match="other files or network"):
                kwargs["io_open"](url, 1, {})
        return original(buffer, **kwargs)

    monkeypatch.setattr(av, "open", audited)
    with media.open_media(audio_file.read_bytes()) as container:
        assert container.streams.audio


def test_recognized_but_corrupt_media_has_clear_error(tmp_path):
    damaged = tmp_path / "damaged.mp4"
    damaged.write_bytes(b"\x00\x00\x00\x20ftyp" + b"\0" * 100)
    with pytest.raises(ValueError, match="could not be opened"):
        inspect(damaged, tmp_path)


def test_duration_bound_rejects_before_generating_preview(video_file, tmp_path, monkeypatch):
    monkeypatch.setattr(media, "MAX_DURATION_SECONDS", 1)
    with pytest.raises(ValueError, match="30 minutes"):
        inspect(video_file, tmp_path)
    assert not (tmp_path / "artifacts").exists()


def test_frame_dimension_bound_rejects_before_decode(video_file, tmp_path, monkeypatch):
    monkeypatch.setattr(media, "MAX_PIXELS", 100)
    with pytest.raises(ValueError, match="dimensions|opened|preview"):
        inspect(video_file, tmp_path)


def test_decode_limit_keeps_available_samples_and_discloses_partial(video_file, tmp_path, monkeypatch):
    monkeypatch.setattr(media, "MAX_DECODED_FRAMES", 2)
    result = inspect(video_file, tmp_path)
    assert result["keyframes"]
    assert len(result["keyframes"]) < 8
    assert any("decoding limit" in warning for warning in result["warnings"])


def test_sampling_deadline_keeps_decoded_frames_and_exports_manifest(video_file, tmp_path, monkeypatch):
    clock = {"now": 0.0}
    original_timestamp = media._timestamp
    original = video_file.read_bytes()

    def timestamp_after_slow_decode(frame):
        value = original_timestamp(frame)
        # The first real frame is useful; subsequent sampling encounters the
        # elapsed deadline. Export must not reuse that expired sampling budget.
        clock["now"] = media.MAX_SECONDS + 1
        return value

    monkeypatch.setattr(media.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(media, "_timestamp", timestamp_after_slow_decode)
    result = inspect(video_file, tmp_path)

    assert len(result["keyframes"]) == 1
    assert "time limit" in result["message"]
    assert len(result["artifacts"]) == 3
    with Image.open(result["contact_sheet"]["path"]) as sheet:
        assert sheet.format == "PNG"
    manifest_artifact = next(item for item in result["artifacts"] if item["format"] == "json")
    manifest = json.loads(Path(manifest_artifact["path"]).read_text())
    assert manifest["warnings"] == result["warnings"]
    assert manifest["frames"][0]["sha256"] == result["keyframes"][0]["sha256"]
    assert video_file.read_bytes() == original


def test_time_limit_prevents_preview_work(video_file, tmp_path, monkeypatch):
    monkeypatch.setattr(media, "MAX_SECONDS", -1)
    with pytest.raises(ValueError, match="time limit"):
        inspect(video_file, tmp_path)


def test_missing_timestamps_and_duration_are_not_invented(audio_file, tmp_path, monkeypatch):
    frames = [av.VideoFrame.from_image(Image.new("RGB", (20, 10), "red")) for _ in range(10)]
    codec = SimpleNamespace(name="fixture", width=20, height=10, options={})
    stream = SimpleNamespace(type="video", index=0, duration=None, time_base=None,
                             start_time=None, codec_context=codec, average_rate=None,
                             disposition=SimpleNamespace(value=0))

    class Container:
        duration = None
        streams = [stream]
        metadata = {}
        format = SimpleNamespace(name="fixture")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def decode(self, _stream):
            yield from frames

        def seek(self, *_args, **_kwargs):
            pytest.fail("unknown duration should not seek")

    monkeypatch.setattr(media, "open_media", lambda _raw: Container())
    result = inspect(audio_file, tmp_path)
    assert len(result["keyframes"]) == 8
    assert result["duration_seconds"] is None
    assert all(item["timestamp"] is None for item in result["keyframes"])
    assert all(item["label"] == "Timestamp unavailable" for item in result["keyframes"])
    assert "Duration is unavailable" in result["message"]
    assert "times are left unknown" in result["message"]


def test_artifact_writer_sanitizes_name_and_never_overwrites(tmp_path):
    result = media.write_artifact(tmp_path / "artifacts", "../../outside", ".txt", b"first")
    again = media.write_artifact(tmp_path / "artifacts", "../../outside", ".txt", b"second")
    assert Path(result["path"]).parent == tmp_path / "artifacts"
    assert result["path"] != again["path"]
    assert Path(result["path"]).read_bytes() == b"first"
    assert result["sha256"] == hashlib.sha256(b"first").hexdigest()


@pytest.mark.parametrize("suffix", ["../outside", ".json/else", "txt", ".exe:stream"])
def test_artifact_suffix_cannot_escape(tmp_path, suffix):
    with pytest.raises(ValueError, match="file extension"):
        media.write_artifact(tmp_path / "artifacts", "name", suffix, b"data")


def test_artifact_limit_applies_before_creating_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "MAX_ARTIFACT_BYTES", 3)
    with pytest.raises(ValueError, match="output limit"):
        media.write_artifact(tmp_path / "artifacts", "name", ".txt", b"four")
    assert not (tmp_path / "artifacts").exists()


def test_symbolic_link_sources_and_artifact_directories_are_rejected(audio_file, tmp_path):
    link = tmp_path / "linked.wav"
    try:
        link.symlink_to(audio_file)
    except OSError:
        pytest.skip("Creating symbolic links is unavailable for this Windows account")
    with pytest.raises(ValueError, match="symbolic link"):
        media.read_media(str(link), [str(link)])
    folder_link = tmp_path / "output-link"
    folder_link.symlink_to(tmp_path / "real-output", target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        media.write_artifact(folder_link, "name", ".txt", b"test")
