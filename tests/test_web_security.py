import asyncio
from contextlib import asynccontextmanager
import socket

import httpx
import pytest

from app.core import guard
from app.core import http


@pytest.mark.parametrize("url", [
    "http://127.0.0.2/", "http://127.0.0.1./", "http://169.254.169.254/",
    "http://[::1]/", "http://[::ffff:127.0.0.1]/", "http://[fd00::1]/",
    "http://100.64.0.1/", "http://224.0.0.1/", "http://2130706433/",
    "http://0177.0.0.1/", "http://0x7f000001/", "http://user:secret@example.com/",
    "file:///etc/passwd", "http://localhost./", "http://example.com:99999/",
    "http://example.com\\@127.0.0.1/", "http://example.com/\nprivate",
])
def test_refuses_unsafe_urls(url):
    assert not guard.validate_safe_url(url)[0]


def test_dns_rejects_mixed_public_private_answers(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [
        (2, 1, 6, "", ("93.184.216.34", 443)), (2, 1, 6, "", ("10.0.0.1", 443)),
    ])
    with pytest.raises(ValueError, match="non-public"):
        guard.resolve_public_addresses("https://example.com")


class RawStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"a" * 70000
        yield b"b" * 70000


class Client:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.cookies = httpx.Cookies()

    @asynccontextmanager
    async def stream(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        try:
            yield self.response
        finally:
            await self.response.aclose()


def test_pins_connection_ip_and_preserves_tls_hostname(monkeypatch):
    monkeypatch.setattr(http, "resolve_public_addresses", lambda url: ["93.184.216.34"])
    client = Client(httpx.Response(200, stream=RawStream()))
    status, _, body = asyncio.run(http._read_public(
        client, "https://example.com:8443/Case", "GET", {"Host": "evil"}, None, 200000,
    ))
    assert status == 200 and len(body) == 140000
    method, url, kwargs = client.calls[0]
    assert str(url) == "https://93.184.216.34:8443/Case"
    assert kwargs["headers"]["Host"] == "example.com:8443"
    assert kwargs["extensions"]["sni_hostname"] == "example.com"


def test_stream_limit_closes_response(monkeypatch):
    monkeypatch.setattr(http, "resolve_public_addresses", lambda url: ["93.184.216.34"])
    client = Client(httpx.Response(200, stream=RawStream()))
    with pytest.raises(ValueError, match="exceeds"):
        asyncio.run(http._read_public(client, "https://example.com", "GET", {}, None, 100))
    assert client.response.is_closed


def test_redirect_to_private_host_is_not_requested(monkeypatch):
    calls = []

    async def fake_read(client, url, *args):
        calls.append(url)
        return 302, {"location": "http://169.254.169.254/metadata"}, b""

    monkeypatch.setattr(http, "_read_public", fake_read)
    result = asyncio.run(http.relai_fetch("https://example.com", cache=False))
    assert not result.ok
    assert calls == ["https://example.com"]


def test_cross_origin_redirect_drops_credentials(monkeypatch):
    calls = []

    async def fake_read(client, url, method, headers, body, limit):
        calls.append(dict(headers))
        if len(calls) == 1:
            return 302, {"location": "https://other.example/page"}, b""
        return 200, {"content-type": "text/plain"}, b"safe"

    monkeypatch.setattr(http, "_read_public", fake_read)
    result = asyncio.run(http.relai_fetch("https://example.com", cache=False,
                                        headers={"Authorization": "secret", "Cookie": "session"}))
    assert result.ok
    assert "Authorization" not in calls[1] and "Cookie" not in calls[1]


def test_server_cookies_do_not_leak_between_sites_sharing_an_ip(monkeypatch):
    calls = []
    original_client = httpx.AsyncClient

    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(302, headers={
                "location": "https://other.example/page",
                "set-cookie": "private_session=secret; Path=/; Secure",
            })
        return httpx.Response(200, stream=RawStream())

    monkeypatch.setattr(http, "resolve_public_addresses", lambda url: ["93.184.216.34"])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(
        transport=httpx.MockTransport(respond), **kwargs,
    ))
    result = asyncio.run(http.relai_fetch("https://example.com", cache=False, retries=0))
    assert result.ok and len(calls) == 2
    assert calls[1].headers["host"] == "other.example"
    assert "cookie" not in calls[1].headers


def test_cache_accounts_for_response_limit_and_returns_copies(monkeypatch):
    http._cache.clear()
    calls = []

    async def fake_read(client, url, method, headers, body, limit):
        calls.append(limit)
        return 200, {}, b"hello"

    monkeypatch.setattr(http, "_read_public", fake_read)
    first = asyncio.run(http.relai_fetch("https://example.com", max_bytes=10))
    first.text = "changed"
    second = asyncio.run(http.relai_fetch("https://example.com", max_bytes=10))
    asyncio.run(http.relai_fetch("https://example.com", max_bytes=20))
    assert second.text == "hello"
    assert calls == [10, 20]
    http._cache.clear()


def test_concurrency_limit_survives_cancellation(monkeypatch):
    active = maximum = 0

    async def fake_read(*args):
        nonlocal active, maximum
        active += 1
        maximum = max(active, maximum)
        try:
            await asyncio.sleep(0.02)
            return 200, {}, b"ok"
        finally:
            active -= 1

    async def run():
        tasks = [asyncio.create_task(http.relai_fetch(f"https://site{i}.example/", cache=False))
                 for i in range(20)]
        await asyncio.sleep(0.01)
        tasks[12].cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        assert (await http.relai_fetch("https://last.example/", cache=False)).ok

    monkeypatch.setattr(http, "_read_public", fake_read)
    asyncio.run(run())
    assert maximum <= http.MAX_CONCURRENT and active == 0


def test_browser_url_is_encoded_as_data(monkeypatch):
    import json
    from app.browser import sessions
    calls = []
    url = "https://example.com/');throw('injected"

    class BrowserClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, endpoint, **kwargs):
            calls.append(kwargs["json"])
            return httpx.Response(200, json={"title": "Safe"})

    monkeypatch.setattr(guard, "resolve_public_addresses", lambda url: ["93.184.216.34"])
    monkeypatch.setattr(sessions.httpx, "AsyncClient", BrowserClient)
    result = asyncio.run(sessions._navigate_via_browser("https://browser.example", url))
    assert result["ok"]
    assert f"page.goto({json.dumps(url)}," in calls[0]["code"]
