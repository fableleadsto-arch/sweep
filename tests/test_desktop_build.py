import hashlib
import json
import subprocess
import sys
from datetime import datetime
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from scripts import build_desktop


@pytest.mark.parametrize("status,dirty", [("", False), (" M private.env\n?? private-file.txt\n", True)])
def test_source_state_reports_dirty_without_exposing_paths(tmp_path, monkeypatch, status, dirty):
    revision = "a" * 40
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=revision if "rev-parse" in command else status)

    monkeypatch.setattr(build_desktop.subprocess, "run", run)
    assert build_desktop.source_state(tmp_path) == {"revision": revision, "dirty": dirty}
    assert calls[1][0][-1] == "--untracked-files=normal"
    assert all(kwargs["cwd"] == tmp_path and kwargs["timeout"] == 10 for _, kwargs in calls)


@pytest.mark.parametrize("failure", [FileNotFoundError(), subprocess.TimeoutExpired("git", 10)])
def test_source_state_does_not_report_clean_when_git_is_unavailable(tmp_path, monkeypatch, failure):
    def run(*args, **kwargs):
        raise failure

    monkeypatch.setattr(build_desktop.subprocess, "run", run)
    assert build_desktop.source_state(tmp_path) == {"revision": None, "dirty": None}


def test_source_state_keeps_revision_when_status_fails(tmp_path, monkeypatch):
    responses = iter([
        SimpleNamespace(returncode=0, stdout="b" * 40),
        SimpleNamespace(returncode=1, stdout=""),
    ])
    monkeypatch.setattr(build_desktop.subprocess, "run", lambda *args, **kwargs: next(responses))
    assert build_desktop.source_state(tmp_path) == {"revision": "b" * 40, "dirty": None}


def test_build_info_records_only_explicit_metadata(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "2.3.4"\n')
    (tmp_path / ".env").write_text("PRIVATE_TOKEN=do-not-include-file-contents")
    monkeypatch.setenv("PRIVATE_TOKEN", "do-not-include-environment")
    monkeypatch.setattr(build_desktop, "source_state", lambda root: {"revision": "c" * 40, "dirty": True})

    def version(name):
        if name == "httpx":
            raise build_desktop.metadata.PackageNotFoundError(name)
        return "1.2.3"

    monkeypatch.setattr(build_desktop.metadata, "version", version)
    artifact = build_desktop.write_build_info(tmp_path, tmp_path)
    raw = artifact.read_text(encoding="utf-8")
    info = json.loads(raw)
    assert info["sweep_version"] == "2.3.4"
    assert info["source"] == {"revision": "c" * 40, "dirty": True}
    assert info["dependencies"]["PySide6-Essentials"] == "1.2.3"
    assert info["dependencies"]["httpx"] is None
    assert set(info["dependencies"]) == set(build_desktop.RECORDED_DISTRIBUTIONS)
    assert datetime.fromisoformat(info["built_at_utc"]).utcoffset().total_seconds() == 0
    assert info["python"]["version"]
    assert set(info["platform"]) == {"system", "machine"}
    assert str(tmp_path) not in raw
    assert "do-not-include" not in raw
    assert ".env" not in raw
    assert not list(tmp_path.glob("*.tmp"))


def test_checksums_identify_and_verify_current_artifacts(tmp_path):
    app = tmp_path / "Sweep"
    app.mkdir()
    executable = app / "Sweep.exe"
    executable.write_bytes(b"application")
    setup = tmp_path / "Sweep-Setup.exe"
    setup.write_bytes(b"installer")
    checksum_file = build_desktop.write_checksums(tmp_path, [executable, setup])
    lines = checksum_file.read_text().splitlines()
    assert len(lines) == 2
    assert [line.split("  ", 1)[1] for line in lines] == ["Sweep-Setup.exe", "Sweep/Sweep.exe"]
    for line in lines:
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == expected
    old_checksum = lines[1].split("  ")[0]
    executable.write_bytes(b"changed application")
    assert hashlib.sha256(executable.read_bytes()).hexdigest() != old_checksum


def test_checksums_reject_artifacts_outside_distribution(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    outside = tmp_path / "private.txt"
    outside.write_text("private")
    with pytest.raises(ValueError):
        build_desktop.write_checksums(dist, [outside])
    assert not (dist / "SHA256SUMS.txt").exists()


@pytest.fixture
def fake_build(tmp_path, monkeypatch):
    monkeypatch.setattr(build_desktop, "ROOT", tmp_path)
    monkeypatch.setattr(build_desktop.sys, "platform", "win32")
    monkeypatch.setattr(build_desktop.metadata, "distributions", lambda: [])
    monkeypatch.setattr(build_desktop.metadata, "version", lambda name: "1.2.3")
    monkeypatch.setattr(build_desktop, "source_state", lambda root: {"revision": "d" * 40, "dirty": True})
    monkeypatch.setitem(sys.modules, "PySide6.QtWidgets", SimpleNamespace(
        QApplication=SimpleNamespace(instance=lambda: object()),
    ))
    monkeypatch.setitem(sys.modules, "sweep.desktop.owl", SimpleNamespace(
        owl_icon=lambda: SimpleNamespace(pixmap=lambda *args: SimpleNamespace(save=lambda *args: True)),
    ))
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/DESKTOP.md").write_text("Desktop instructions")
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "2.0.0"\n')
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist/SHA256SUMS.txt").write_text("stale checksums")
    (tmp_path / "dist/Sweep-Setup.exe").write_bytes(b"old setup")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        name = command[command.index("--name") + 1]
        if name == "Sweep":
            app = tmp_path / "dist/Sweep"
            app.mkdir()
            (app / "Sweep.exe").write_bytes(b"current app")
        else:
            with ZipFile(tmp_path / "build/desktop/payload.zip") as archive:
                assert json.loads(archive.read("build-info.json"))["source"]["dirty"] is True
            (tmp_path / "dist/Sweep-Setup.exe").write_bytes(b"current setup")

    monkeypatch.setattr(build_desktop.subprocess, "run", run)
    return tmp_path, calls


@pytest.mark.parametrize("app_only", [False, True])
def test_build_writes_metadata_before_packaging_and_checksums_only_current_outputs(fake_build, app_only):
    root, calls = fake_build
    assert build_desktop.main(["--app-only"] if app_only else []) == 0
    assert len(calls) == (1 if app_only else 2)
    checksums = (root / "dist/SHA256SUMS.txt").read_text()
    assert "Sweep/Sweep.exe" in checksums
    assert "Sweep/build-info.json" in checksums
    assert ("Sweep-Setup.exe" in checksums) is (not app_only)
    if app_only:
        assert (root / "dist/Sweep-Setup.exe").read_bytes() == b"old setup"


def test_failed_build_removes_stale_checksums(fake_build, monkeypatch):
    root, _ = fake_build

    def run(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(build_desktop.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        build_desktop.main([])
    assert not (root / "dist/SHA256SUMS.txt").exists()
