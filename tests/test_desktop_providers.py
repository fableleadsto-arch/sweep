import asyncio
import json
import sys

import pytest

from sweep.desktop import providers


@pytest.fixture(autouse=True)
def no_live_configuration(monkeypatch):
    for key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OLLAMA_MODEL", "PROVIDER_ORDER", "OLLAMA_BASE_URL"):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.skipif(sys.platform != "win32", reason="DPAPI is Windows-only")
def test_keys_are_protected_and_presence_flags_do_not_reveal_them(tmp_path):
    secret = "sweep-test-credential-not-a-real-key"
    value = providers.save_config(tmp_path, {"provider": "openai", "model": "example-model"}, {"openai": secret})
    raw = (tmp_path / providers.CONFIG_NAME).read_text()
    assert secret not in raw
    assert secret not in json.dumps(value)
    assert value["key_present"]["openai"]
    assert providers.resolve_settings(tmp_path).openai_api_key == secret
    providers.save_config(tmp_path, {}, {"openai": ""})
    assert not providers.load_config(tmp_path)["key_present"]["openai"]


@pytest.mark.parametrize("url", ["ftp://localhost", "http://user:secret@localhost", "http://localhost/?key=secret", "http://localhost\n"])
def test_provider_address_rejects_credentials_and_invalid_urls(tmp_path, url):
    with pytest.raises(ValueError):
        providers.save_config(tmp_path, {"ollama_base_url": url})


def test_installed_ollama_model_replaces_missing_legacy_default(tmp_path, monkeypatch):
    providers.save_config(tmp_path, {"provider": "ollama"})

    async def models(url):
        return {"models": ["local-small:3b", "local-large:7b"], "message": "Ready"}

    seen = []
    class FakeChain:
        def __init__(self, settings):
            seen.append(settings)
        async def generate(self, **kwargs):
            return providers.ProviderResult(text="hello", model="local-small:3b", provider="ollama")
    monkeypatch.setattr(providers, "_ollama_models", models)
    monkeypatch.setattr(providers, "ProviderChain", FakeChain)
    result = asyncio.run(providers.generate_chat(tmp_path, system="hello", messages=[]))
    assert result.text == "hello"
    assert seen[0].ollama_model == "local-small:3b"
    assert seen[0].provider_order == ["ollama"]


def test_explicit_missing_model_does_not_silently_use_another(tmp_path, monkeypatch):
    providers.save_config(tmp_path, {"provider": "ollama", "model": "missing:7b"})
    async def models(url):
        return {"models": ["installed:3b"], "message": "Ready"}
    monkeypatch.setattr(providers, "_ollama_models", models)
    with pytest.raises(RuntimeError, match="not installed"):
        asyncio.run(providers.generate_chat(tmp_path, system="test", messages=[]))


def test_no_key_is_reported_before_cloud_request(tmp_path):
    providers.save_config(tmp_path, {"provider": "openai"})
    with pytest.raises(RuntimeError, match="API key"):
        asyncio.run(providers.generate_chat(tmp_path, system="test", messages=[]))


def test_remote_model_discovery_does_not_make_unsolicited_request(monkeypatch):
    monkeypatch.setattr(providers.httpx, "AsyncClient", lambda **kwargs: pytest.fail("Remote discovery attempted"))
    result = asyncio.run(providers._ollama_models("https://models.example.com"))
    assert result["checked"] is False


def test_start_server_rejects_remote_address_before_launch(tmp_path, monkeypatch):
    providers.save_config(tmp_path, {"ollama_base_url": "https://models.example.com"})
    monkeypatch.setattr(providers.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Remote launch attempted"))
    with pytest.raises(RuntimeError, match="only a local HTTP"):
        asyncio.run(providers.start_local_server(tmp_path))


def test_start_server_uses_installed_executable_and_loopback(tmp_path, monkeypatch):
    local = tmp_path / "local"
    executable = local / "Programs/Ollama/ollama.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"test executable")
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.setattr(providers.shutil, "which", lambda name: None)
    providers.save_config(tmp_path, {"ollama_base_url": "http://localhost:11434"})
    replies = iter([False, True])

    async def models(url):
        return {"reachable": next(replies), "models": []}

    async def sleep(delay):
        pass

    calls = []
    monkeypatch.setattr(providers, "_ollama_models", models)
    monkeypatch.setattr(providers.asyncio, "sleep", sleep)
    monkeypatch.setattr(providers.subprocess, "Popen", lambda command, **kwargs: calls.append((command, kwargs)))
    result = asyncio.run(providers.start_local_server(tmp_path))
    assert result["ollama"]["reachable"]
    assert calls[0][0] == [str(executable), "serve"]
    assert calls[0][1]["env"]["OLLAMA_HOST"] == "127.0.0.1:11434"
    assert calls[0][1]["stdout"] == providers.subprocess.DEVNULL


def test_start_server_does_not_restart_existing_server(tmp_path, monkeypatch):
    async def models(url):
        return {"reachable": True, "models": ["installed:2b"]}
    monkeypatch.setattr(providers, "_ollama_models", models)
    monkeypatch.setattr(providers.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Duplicate server launch"))
    result = asyncio.run(providers.start_local_server(tmp_path))
    assert result["message"] == "Ollama is already running."


def test_local_chat_bounds_context_and_allows_cpu_inference_time(tmp_path, monkeypatch):
    providers.save_config(tmp_path, {"provider": "ollama", "model": "installed:3b"})
    seen = {}
    async def models(url):
        return {"models": ["installed:3b"]}
    class FakeChain:
        def __init__(self, settings):
            seen["timeout"] = settings.request_timeout_seconds
        async def generate(self, **kwargs):
            seen.update(kwargs)
            return providers.ProviderResult(text="Hello", model="installed:3b", provider="ollama")
    monkeypatch.setattr(providers, "_ollama_models", models)
    monkeypatch.setattr(providers, "ProviderChain", FakeChain)
    messages = [{"role": "assistant", "content": "long source " * 4000} for _ in range(12)]
    messages.append({"role": "user", "content": "My latest question"})
    asyncio.run(providers.generate_chat(tmp_path, system="test", messages=messages))
    assert sum(len(item["content"]) for item in seen["messages"]) <= 10000
    assert seen["messages"][-1] == messages[-1]
    assert seen["timeout"] == 180
    assert seen["max_tokens"] == 600
