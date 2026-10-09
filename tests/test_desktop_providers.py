"""Desktop inference verifies downloaded local models before sending user content."""
import asyncio
import json

import httpx
import pytest

from sweep.desktop import providers


@pytest.fixture(autouse=True)
def no_live_configuration(monkeypatch):
    for key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OLLAMA_MODEL", "PROVIDER_ORDER", "OLLAMA_BASE_URL"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def local_http(monkeypatch):
    original = httpx.AsyncClient
    calls = []

    def install(handler):
        def client(**kwargs):
            calls.append(kwargs)
            return original(transport=httpx.MockTransport(handler), **kwargs)
        monkeypatch.setattr(providers.httpx, "AsyncClient", client)
        return calls

    return install


def test_old_cloud_preferences_migrate_without_reading_credentials(tmp_path):
    protected = {"openai": "old-encrypted-blob-not-a-real-key"}
    (tmp_path / providers.CONFIG_NAME).write_text(json.dumps({
        "schema_version": 1, "provider": "openai", "model": "cloud-chat",
        "vision_provider": "gemini", "vision_model": "cloud-vision", "protected_keys": protected,
    }))
    config = providers.load_config(tmp_path)
    assert config["provider"] == config["vision_provider"] == "ollama"
    assert config["model"] == config["vision_model"] == ""
    assert config["local_only"] is True
    assert not any(config["key_present"].values())
    assert "old-encrypted" not in json.dumps(config)
    providers.save_config(tmp_path, {"model": "downloaded:3b"})
    assert json.loads((tmp_path / providers.CONFIG_NAME).read_text())["protected_keys"] == protected


def test_environment_and_dotenv_cloud_credentials_are_ignored(tmp_path, monkeypatch):
    for key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(key, "environment-test-secret")
    monkeypatch.setenv("PROVIDER_ORDER", '["openai", "gemini", "anthropic"]')
    (tmp_path / ".env").write_text("OPENAI_API_KEY=dotenv-test-secret\n")
    settings = providers.resolve_settings(tmp_path)
    assert settings.provider_order == ["ollama"]
    assert settings.openai_api_key == settings.gemini_api_key == settings.anthropic_api_key == ""
    assert not any(providers.load_config(tmp_path)["key_present"].values())


@pytest.mark.parametrize("config,keys", [
    ({"provider": "openai"}, None), ({"vision_provider": "gemini"}, None),
    ({}, {"anthropic": "test-secret"}),
])
def test_cloud_configuration_is_rejected_before_saving(tmp_path, config, keys):
    with pytest.raises(ValueError, match="Cloud providers and API keys are disabled"):
        providers.save_config(tmp_path, config, keys)
    assert not (tmp_path / providers.CONFIG_NAME).exists()


@pytest.mark.parametrize("url", [
    "ftp://localhost", "https://localhost:11434", "http://user:secret@localhost",
    "http://localhost/?key=secret", "http://localhost\n", "http://localhost/custom",
    "http://models.example.com", "http://192.168.1.10:11434", "http://127.0.0.1:99999",
])
def test_provider_address_rejects_nonlocal_or_ambiguous_endpoints(tmp_path, url):
    with pytest.raises(ValueError):
        providers.save_config(tmp_path, {"ollama_base_url": url})


@pytest.mark.parametrize("url,expected", [
    ("http://localhost:11434/", "http://127.0.0.1:11434"),
    ("http://127.0.0.1:12345", "http://127.0.0.1:12345"),
    ("http://[::1]:11434", "http://[::1]:11434"),
])
def test_local_endpoint_is_canonicalized(tmp_path, url, expected):
    assert providers.save_config(tmp_path, {"ollama_base_url": url})["ollama_base_url"] == expected


@pytest.mark.parametrize("model", ["model:cloud", "cloud/model", "model-cloud:latest"])
def test_cloud_model_names_are_not_saved(tmp_path, model):
    with pytest.raises(ValueError, match="Cloud models are not allowed"):
        providers.save_config(tmp_path, {"model": model})


def test_installed_model_replaces_missing_legacy_default(tmp_path, monkeypatch):
    providers.save_config(tmp_path, {"vision_model": "vision-small:2b"})

    async def models(url):
        return {"models": ["vision-small:2b", "local-small:3b", "local-large:7b"], "message": "Ready"}

    verified = []

    async def verify(url, model):
        verified.append(model)

    seen = []

    class FakeChain:
        def __init__(self, settings):
            seen.append(settings)

        async def generate(self, **kwargs):
            assert verified == ["local-small:3b"]
            return providers.ProviderResult(text="hello", model="local-small:3b", provider="ollama")

    monkeypatch.setattr(providers, "_ollama_models", models)
    monkeypatch.setattr(providers, "require_local_model", verify)
    monkeypatch.setattr(providers, "ProviderChain", FakeChain)
    result = asyncio.run(providers.generate_chat(tmp_path, system="hello", messages=[]))
    assert result.text == "hello"
    assert seen[0].ollama_model == "local-small:3b"
    assert seen[0].provider_order == ["ollama"]


def test_explicit_missing_model_does_not_silently_use_another(tmp_path, monkeypatch):
    providers.save_config(tmp_path, {"model": "missing:7b"})

    async def models(url):
        return {"models": ["installed:3b"], "message": "Ready"}

    monkeypatch.setattr(providers, "_ollama_models", models)
    monkeypatch.setattr(providers, "ProviderChain", lambda *_: pytest.fail("Missing model reached inference"))
    with pytest.raises(RuntimeError, match="not installed"):
        asyncio.run(providers.generate_chat(tmp_path, system="test", messages=[]))


def test_remote_model_discovery_does_not_make_unsolicited_request(monkeypatch):
    monkeypatch.setattr(providers.httpx, "AsyncClient", lambda **kwargs: pytest.fail("Remote discovery attempted"))
    result = asyncio.run(providers._ollama_models("https://models.example.com"))
    assert result["checked"] is False


def test_discovery_filters_cloud_aliases_and_sorts_local_models(local_http):
    def respond(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [
            {"name": "large:7b", "size": 700}, {"name": "small:2b", "size": 200},
            {"name": "small:2b", "size": 200}, {"name": "hidden-alias", "remote_host": "https://example.com"},
            {"name": "another-alias", "remote_model": "private"}, {"name": "model:cloud"},
            {"name": "text-embed:latest"}, {"name": None},
        ]})
    calls = local_http(respond)
    result = asyncio.run(providers._ollama_models("http://127.0.0.1:11434"))
    assert result["models"] == ["small:2b", "large:7b"]
    assert all(call["trust_env"] is False and call["follow_redirects"] is False for call in calls)


def test_start_server_rejects_legacy_remote_address_before_launch(tmp_path, monkeypatch):
    (tmp_path / providers.CONFIG_NAME).write_text(json.dumps({
        "schema_version": 1, "ollama_base_url": "https://models.example.com",
    }))
    monkeypatch.setattr(providers.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Remote launch attempted"))
    with pytest.raises(ValueError, match="HTTP loopback"):
        asyncio.run(providers.start_local_server(tmp_path))


def test_start_server_uses_installed_executable_with_cloud_disabled(tmp_path, monkeypatch):
    local = tmp_path / "local"
    executable = local / "Programs/Ollama/ollama.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"test executable")
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.setenv("OLLAMA_NO_CLOUD", "0")
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
    assert calls[0][1]["env"]["OLLAMA_NO_CLOUD"] == "1"
    assert calls[0][1]["env"]["OLLAMA_KEEP_ALIVE"] == "1m"
    assert calls[0][1]["stdout"] == providers.subprocess.DEVNULL


def test_start_server_does_not_restart_existing_server(tmp_path, monkeypatch):
    async def models(url):
        return {"reachable": True, "models": ["installed:2b"]}

    monkeypatch.setattr(providers, "_ollama_models", models)
    monkeypatch.setattr(providers.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Duplicate server launch"))
    result = asyncio.run(providers.start_local_server(tmp_path))
    assert result["message"] == "Local intelligence is already running."


@pytest.mark.parametrize("info", [
    {"model_info": {"general.architecture": "qwen"}, "remote_host": "https://cloud.example.com"},
    {"model_info": {"general.architecture": "qwen"}, "remote_model": "remote-private"},
    {}, {"model_info": {}}, {"model_info": "not metadata"}, {"model_info": ["not metadata"]}, [],
])
def test_unverified_or_cloud_alias_never_receives_chat_content(tmp_path, local_http, info):
    providers.save_config(tmp_path, {"model": "apparently-local:latest"})
    sent = []

    def respond(request):
        sent.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "apparently-local:latest"}]})
        assert request.url.path == "/api/show", "User content reached unverified inference"
        assert json.loads(request.content) == {"model": "apparently-local:latest"}
        return httpx.Response(200, json=info)

    local_http(respond)
    with pytest.raises(RuntimeError, match="could not verify"):
        asyncio.run(providers.generate_chat(tmp_path, system="private system", messages=[{"role": "user", "content": "private message"}]))
    assert len(sent) == 2
    assert all(b"private message" not in request.content for request in sent)


@pytest.mark.parametrize("info", [
    {"model_info": {"general.architecture": "qwen"}, "capabilities": ["completion"]},
    {"model_info": {"general.architecture": "qwen"}, "capabilities": None},
    {"model_info": {"general.architecture": "qwen"}, "capabilities": "vision"},
])
def test_vision_requires_verified_image_capability(local_http, info):
    local_http(lambda request: httpx.Response(200, json=info))
    with pytest.raises(RuntimeError):
        asyncio.run(providers.require_local_model("http://127.0.0.1:11434", "local:2b", vision=True))


@pytest.mark.parametrize("status", [301, 401, 404, 500])
def test_model_verification_does_not_follow_redirects_or_expose_server_errors(local_http, status):
    calls = local_http(lambda request: httpx.Response(status, headers={"location": "https://cloud.example.com"}, text="private server detail"))
    with pytest.raises(RuntimeError, match="no content was sent") as caught:
        asyncio.run(providers.require_local_model("http://127.0.0.1:11434", "local:2b"))
    assert "private server detail" not in str(caught.value)
    assert calls[0]["follow_redirects"] is False
    assert calls[0]["trust_env"] is False


@pytest.mark.parametrize("base_url,model", [
    ("http://models.example.com", "local:2b"), ("https://localhost:11434", "local:2b"),
    ("http://127.0.0.1:11434", "renamed-cloud-model"),
])
def test_nonlocal_models_are_rejected_without_network(monkeypatch, base_url, model):
    monkeypatch.setattr(providers.httpx, "AsyncClient", lambda **kwargs: pytest.fail("Nonlocal verification attempted"))
    with pytest.raises(RuntimeError, match="downloaded model"):
        asyncio.run(providers.require_local_model(base_url, model))


def test_local_chat_verifies_model_then_generates_without_cloud_fallback(tmp_path, local_http, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-environment-secret")
    providers.save_config(tmp_path, {"model": "installed:3b"})
    sent = []

    def respond(request):
        sent.append(request)
        assert request.url.host == "127.0.0.1"
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "installed:3b"}]})
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"model_info": {"general.architecture": "qwen"}})
        assert request.url.path == "/api/generate"
        assert [request.url.path for request in sent] == ["/api/tags", "/api/show", "/api/generate"]
        return httpx.Response(500, text="server failure with private detail")

    calls = local_http(respond)
    with pytest.raises(RuntimeError, match="Local intelligence could not finish"):
        asyncio.run(providers.generate_chat(tmp_path, system="test", messages=[{"role": "user", "content": "hello"}]))
    assert len(sent) == 3
    assert all(call["trust_env"] is False and call["follow_redirects"] is False for call in calls)


def test_local_chat_bounds_context_and_allows_cpu_inference_time(tmp_path, monkeypatch):
    providers.save_config(tmp_path, {"model": "installed:3b"})
    seen = {}

    async def models(url):
        return {"models": ["installed:3b"]}

    async def verify(url, model):
        pass

    class FakeChain:
        def __init__(self, settings):
            seen["timeout"] = settings.request_timeout_seconds

        async def generate(self, **kwargs):
            seen.update(kwargs)
            return providers.ProviderResult(text="Hello", model="installed:3b", provider="ollama")

    monkeypatch.setattr(providers, "_ollama_models", models)
    monkeypatch.setattr(providers, "require_local_model", verify)
    monkeypatch.setattr(providers, "ProviderChain", FakeChain)
    messages = [{"role": "assistant", "content": "long source " * 4000} for _ in range(12)]
    messages.append({"role": "user", "content": "My latest question"})
    asyncio.run(providers.generate_chat(tmp_path, system="test", messages=messages))
    assert sum(len(item["content"]) for item in seen["messages"]) <= 10000
    assert seen["messages"][-1] == messages[-1]
    assert seen["timeout"] == 180
    assert seen["max_tokens"] == 600
