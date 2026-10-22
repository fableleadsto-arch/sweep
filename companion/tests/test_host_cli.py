from __future__ import annotations

import json

import pytest

from companion.hosting.__main__ import main


def test_local_admin_workflow_does_not_require_a_service(tmp_path, capsys):
    prefix = ["--state", str(tmp_path / "identity.sqlite")]
    assert main([*prefix, "organization", "Example"]) == 0
    org = capsys.readouterr().out.strip()
    assert main([*prefix, "user", org, "Alice"]) == 0
    user = capsys.readouterr().out.strip()
    assert main([*prefix, "invite", user]) == 0
    invite = capsys.readouterr().out.strip()
    assert invite.startswith("swp_i_") and len(invite) == 49
    assert main([*prefix, "devices", user]) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert main([*prefix, "disable", user]) == 0
    assert json.loads(capsys.readouterr().out) == {"disabled": True}


@pytest.mark.parametrize("options", [
    ["--bind", "0.0.0.0"], ["--bind", "192.0.2.1"], ["--bind", "example.com"],
    ["--port", "0"], ["--port", "65536"], ["--certfile", "missing"],
    ["--allow-host", "*.example.com"],
])
def test_invalid_serve_configuration_does_not_create_state_or_listen(tmp_path, options):
    path = tmp_path / "identity.sqlite"
    with pytest.raises(SystemExit) as exc:
        main(["--state", str(path), "serve", "--config-root", str(tmp_path), *options])
    assert exc.value.code == 2
    assert not path.exists()


def test_serve_uses_one_worker_and_does_not_trust_forwarded_headers(tmp_path, monkeypatch):
    import uvicorn

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **options: calls.append(options))
    assert main(["--state", str(tmp_path / "identity.sqlite"), "serve", "--config-root", str(tmp_path)]) == 0
    assert calls[0]["host"] == "127.0.0.1"
    assert calls[0]["workers"] == 1
    assert calls[0]["proxy_headers"] is False
    assert calls[0]["access_log"] is False
    assert calls[0]["limit_concurrency"] == 64
