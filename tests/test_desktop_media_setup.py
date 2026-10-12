"""Speech setup is explicit, integrity-checked, bounded and non-destructive."""
import hashlib
import io
import json
import stat
import zipfile
from pathlib import Path

import httpx
import pytest

from sweep.desktop import media_setup as setup


def archive_bytes(extra=(), *, omit=()):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(setup.REQUIRED_FILES - set(omit)):
            archive.writestr(f"{setup.MODEL_NAME}/{name}", f"fixture {name}".encode())
        for name, value in extra:
            if isinstance(name, str):
                # ZipInfo's Windows constructor normalizes backslashes, but an
                # untrusted archive produced elsewhere need not do so.
                info = zipfile.ZipInfo("placeholder")
                info.filename = name
                name = info
            archive.writestr(name, value)
    return output.getvalue()


@pytest.fixture
def network(monkeypatch):
    original = httpx.Client
    calls = []

    def install(body=None, *, status=200, headers=None, handler=None, pin=True):
        if body is None:
            body = archive_bytes()
        if pin:
            monkeypatch.setattr(setup, "MODEL_SHA256", hashlib.sha256(body).hexdigest())

        def default(request):
            calls.append(request)
            assert str(request.url) == setup.MODEL_URL
            return httpx.Response(status, content=body, headers=headers)

        def client(**kwargs):
            assert kwargs["trust_env"] is False
            assert kwargs["follow_redirects"] is False
            return original(transport=httpx.MockTransport(handler or default), **kwargs)

        monkeypatch.setattr(setup.httpx, "Client", client)
        return calls

    return install


def test_readiness_does_not_create_storage_load_models_or_download(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Readiness must stay offline")

    monkeypatch.setattr(setup.httpx, "Client", forbidden)
    root = tmp_path / "not-created"
    assert setup.speech_model_path(root) == root / "models" / "speech" / setup.MODEL_NAME
    assert not setup.model_ready(root)
    assert not root.exists()


def test_complete_install_has_owned_manifest_and_reuses_valid_asset(tmp_path, network):
    calls = network()
    progress = []
    first = setup.install_speech_model(tmp_path, progress.append)
    target = setup.speech_model_path(tmp_path)
    assert first["status"] == "ready" and not first["already_installed"]
    assert setup.model_ready(tmp_path)
    record = json.loads((target / setup.MANIFEST_NAME).read_text())
    assert record["archive_sha256"] == setup.MODEL_SHA256
    assert set(record["files"]) == setup.REQUIRED_FILES
    assert progress and "40 MB" in progress[0]
    assert "vosk" not in first["message"].lower()
    assert list(target.parent.iterdir()) == [target]
    before = {p: p.read_bytes() for p in target.rglob("*") if p.is_file()}
    assert setup.install_speech_model(tmp_path)["already_installed"]
    assert len(calls) == 1
    assert all(p.read_bytes() == value for p, value in before.items())


@pytest.mark.parametrize("damage", ["missing", "changed_size", "extra", "manifest", "pin"])
def test_modified_model_is_not_ready_and_is_preserved(tmp_path, network, monkeypatch, damage):
    network()
    setup.install_speech_model(tmp_path)
    target = setup.speech_model_path(tmp_path)
    model = target / "am" / "final.mdl"
    if damage == "missing":
        model.unlink()
    elif damage == "changed_size":
        model.write_bytes(b"changed")
    elif damage == "extra":
        (target / "user-notes.txt").write_text("preserve this file")
    elif damage == "manifest":
        (target / setup.MANIFEST_NAME).write_text("invalid")
    else:
        monkeypatch.setattr(setup, "MODEL_SHA256", "f" * 64)
    before = {p: p.read_bytes() for p in target.rglob("*") if p.is_file()}
    assert not setup.model_ready(tmp_path)
    with pytest.raises(ValueError, match="preserved"):
        setup.install_speech_model(tmp_path)
    assert all(p.read_bytes() == value for p, value in before.items())


def test_existing_invalid_directory_never_downloads_or_overwrites(tmp_path, monkeypatch):
    target = setup.speech_model_path(tmp_path)
    target.mkdir(parents=True)
    note = target / "personal-notes.txt"
    note.write_text("keep")
    monkeypatch.setattr(setup.httpx, "Client", lambda **kwargs: pytest.fail("download attempted"))
    with pytest.raises(ValueError, match="preserved"):
        setup.install_speech_model(tmp_path)
    assert note.read_text() == "keep"


def test_missing_pin_fails_before_download_or_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "MODEL_SHA256", "")
    with pytest.raises(RuntimeError, match="checksum"):
        setup.install_speech_model(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_checksum_mismatch_is_cleaned_and_never_installed(tmp_path, network):
    network(pin=False)
    with pytest.raises(ValueError, match="integrity"):
        setup.install_speech_model(tmp_path)
    assert list((tmp_path / "models" / "speech").iterdir()) == []


@pytest.mark.parametrize("status", [301, 302, 307, 401, 404, 500])
def test_redirects_and_errors_do_not_follow_or_install(tmp_path, network, status):
    calls = network(status=status, headers={"Location": "https://other.invalid/asset.zip"})
    with pytest.raises(ValueError, match="retrieved"):
        setup.install_speech_model(tmp_path)
    assert len(calls) == 1
    assert not setup.model_ready(tmp_path)
    assert list((tmp_path / "models" / "speech").iterdir()) == []


@pytest.mark.parametrize("length", ["-1", "0", "60000001", "invalid"])
def test_declared_size_is_bounded(tmp_path, network, length):
    network(headers={"Content-Length": length})
    with pytest.raises(ValueError, match="size"):
        setup.install_speech_model(tmp_path)
    assert list((tmp_path / "models" / "speech").iterdir()) == []


def test_streamed_download_is_bounded_without_content_length(tmp_path, network, monkeypatch):
    body = archive_bytes()

    def response(request):
        return httpx.Response(200, stream=httpx.ByteStream(body))

    network(body, handler=response)
    monkeypatch.setattr(setup, "MAX_DOWNLOAD_BYTES", 10)
    with pytest.raises(ValueError, match="size limit"):
        setup.install_speech_model(tmp_path)
    assert list((tmp_path / "models" / "speech").iterdir()) == []


def test_interrupted_stream_cleans_only_own_stage(tmp_path, network):
    parent = tmp_path / "models" / "speech"
    parent.mkdir(parents=True)
    existing = parent / ".sweep-speech-unrelated"
    existing.mkdir()
    note = existing / "keep.txt"
    note.write_text("keep")

    class BrokenStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"partial"
            raise httpx.ReadError("interrupted")

    network(handler=lambda request: httpx.Response(200, stream=BrokenStream()))
    with pytest.raises(ValueError, match="interrupted"):
        setup.install_speech_model(tmp_path)
    assert list(parent.iterdir()) == [existing]
    assert note.read_text() == "keep"


def test_download_has_total_time_budget(tmp_path, network, monkeypatch):
    network()
    values = iter([0, setup.DOWNLOAD_SECONDS + 1])
    monkeypatch.setattr(setup.time, "monotonic", lambda: next(values, 1000))
    with pytest.raises(TimeoutError, match="timed out"):
        setup.install_speech_model(tmp_path)
    assert list((tmp_path / "models" / "speech").iterdir()) == []


@pytest.mark.parametrize("path", [
    "../escape", "/outside", "another-model/file", "C:/outside",
    f"{setup.MODEL_NAME}/../escape", f"{setup.MODEL_NAME}/am/../../escape",
    f"{setup.MODEL_NAME}\\am\\final.mdl", f"{setup.MODEL_NAME}/am/final.mdl:stream",
    f"{setup.MODEL_NAME}/am/CON", f"{setup.MODEL_NAME}/am/final.mdl ",
    f"{setup.MODEL_NAME}/./am/final.mdl", f"{setup.MODEL_NAME}//am/final.mdl",
])
def test_unsafe_archive_paths_do_not_escape_or_install(tmp_path, network, path):
    network(archive_bytes([(path, b"unsafe")]))
    with pytest.raises(ValueError, match="unsafe path"):
        setup.install_speech_model(tmp_path)
    assert list((tmp_path / "models" / "speech").iterdir()) == []
    assert not (tmp_path / "escape").exists()


def test_duplicate_case_insensitive_paths_are_rejected(tmp_path, network):
    network(archive_bytes([(f"{setup.MODEL_NAME}/AM/FINAL.MDL", b"duplicate")]))
    with pytest.raises(ValueError, match="duplicate"):
        setup.install_speech_model(tmp_path)


def test_symlink_archive_member_is_rejected(tmp_path, network):
    link = zipfile.ZipInfo(f"{setup.MODEL_NAME}/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    network(archive_bytes([(link, b"../../private")]))
    with pytest.raises(ValueError, match="linked or special"):
        setup.install_speech_model(tmp_path)


@pytest.mark.parametrize("limit,value", [("MAX_ARCHIVE_ENTRIES", 2), ("MAX_EXTRACTED_BYTES", 2)])
def test_archive_entry_and_expansion_budgets(tmp_path, network, monkeypatch, limit, value):
    network()
    monkeypatch.setattr(setup, limit, value)
    with pytest.raises(ValueError, match="too many|extraction limit"):
        setup.install_speech_model(tmp_path)
    assert list((tmp_path / "models" / "speech").iterdir()) == []


def test_missing_required_file_is_not_installed(tmp_path, network):
    network(archive_bytes(omit={"am/final.mdl"}))
    with pytest.raises(ValueError, match="incomplete"):
        setup.install_speech_model(tmp_path)
    assert not setup.model_ready(tmp_path)


def test_target_created_during_download_is_preserved(tmp_path, network):
    body = archive_bytes()
    target = setup.speech_model_path(tmp_path)

    def response(request):
        target.mkdir()
        (target / "keep.txt").write_text("keep")
        return httpx.Response(200, content=body)

    network(body, handler=response)
    with pytest.raises(ValueError, match="existing files were preserved"):
        setup.install_speech_model(tmp_path)
    assert (target / "keep.txt").read_text() == "keep"
    assert not (target / "am").exists()
    assert list(target.parent.iterdir()) == [target]


def test_network_storage_is_rejected_before_access(monkeypatch):
    monkeypatch.setattr(Path, "resolve", lambda *args, **kwargs: pytest.fail("network path resolved"))
    with pytest.raises(ValueError, match="this computer"):
        setup.speech_model_path(r"\\server\share")
    assert not setup.model_ready(r"\\server\share")


def test_redirected_storage_is_rejected_and_target_preserved(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    try:
        (root / "models").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Windows account cannot create symlink fixtures")
    assert not setup.model_ready(root)
    with pytest.raises(ValueError, match="redirect"):
        setup.install_speech_model(root)
    assert list(outside.iterdir()) == []
