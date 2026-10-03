import subprocess

import pytest

import setup_sweep
from sweep import launcher


def test_setup_installs_in_local_venv_without_native_build(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_sweep, "ROOT", tmp_path)
    created, commands = [], []
    monkeypatch.setattr(setup_sweep.venv.EnvBuilder, "create", lambda self, path: created.append(path))
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: commands.append(command))
    assert setup_sweep.main(["--install-only", "--extras", "science"]) == 0
    assert created == [tmp_path / ".venv"]
    assert commands[0][-1] == str(tmp_path) + "[desktop,science]"
    assert "-e" in commands[0]


def test_setup_reports_install_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_sweep, "ROOT", tmp_path)
    monkeypatch.setattr(setup_sweep.venv.EnvBuilder, "create", lambda *args: None)

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "pip")

    monkeypatch.setattr(subprocess, "run", fail)
    assert setup_sweep.main(["--install-only"]) == 1


def test_export_never_overwrites(tmp_path):
    target = tmp_path / "output.json"
    target.write_text("original")
    with pytest.raises(FileExistsError):
        launcher._emit({"new": True}, str(target))
    assert target.read_text() == "original"


def test_controller_cli_reports_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("SWEEP_CONTROLLER_DIR", str(tmp_path))
    from sweep.__main__ import main
    assert main(["--no-llm", "calculate 1/0"]) == 1


def test_scrape_exports_actual_extracted_content(tmp_path, monkeypatch):
    from app.core.http import FetchResult
    from app.core import http

    async def fetch(*args, **kwargs):
        return FetchResult(ok=True, status=200, url="https://example.com/",
                           text="<html><title>Example</title><body><h1>Hello Sweep</h1><p>Public test content.</p></body></html>")

    monkeypatch.setattr(http, "relai_fetch", fetch)
    output = tmp_path / "page.json"
    assert launcher.main(["scrape", "https://example.com/", "--output", str(output)]) == 0
    assert "Hello Sweep" in output.read_text(encoding="utf-8")


def test_server_generates_token_and_cleans_up(tmp_path, monkeypatch):
    import uvicorn
    from app.config import get_settings
    monkeypatch.setenv("SWEEP_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("SWEEP_API_TOKEN", "")
    get_settings.cache_clear()
    seen = []

    def serve(module, **kwargs):
        seen.append(module)
        assert kwargs["host"] == "127.0.0.1"
        token = next(tmp_path.glob("*.token")).read_text()
        assert len(token) >= 32
        assert get_settings().sweep_api_token == token

    monkeypatch.setattr(uvicorn, "run", serve)
    assert launcher.main(["serve", "web"]) == 0
    assert seen == ["app.main:app"]
    assert not list(tmp_path.glob("*.token"))


def test_server_port_validation_and_compact_help(capsys):
    with pytest.raises(SystemExit) as exc:
        launcher.main(["serve", "--port", "0"])
    assert exc.value.code == 2
    assert "between 1 and 65535" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        launcher.main(["serve", "--help"])
    assert exc.value.code == 0
    assert len(capsys.readouterr().out) < 2000
