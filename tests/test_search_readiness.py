import asyncio
import base64
import json
import time
from types import SimpleNamespace

import pytest

from app.search import engine


@pytest.fixture(autouse=True)
def isolated_search(monkeypatch):
    from app import config
    settings = SimpleNamespace(tavily_api_key="", exa_api_key="", jina_api_key="", searxng_base_url="")
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(engine, "_circuit_broken_until", {})
    monkeypatch.setattr(engine, "_last_bing_call", 0)
    return settings


def fetched(text="", *, status=200, blocked=False):
    return SimpleNamespace(ok=200 <= status < 300 and not blocked, status=status, text=text,
                           blocked=blocked, error=None)


def test_bing_returns_real_destination_instead_of_display_citation():
    url = "https://docs.python.org/3/tutorial/index.html"
    encoded = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
    html = f'<li class="b_algo"><cite>https://docs.python.org / tutorial</cite><h2><a href="https://www.bing.com/ck/a?x=1&amp;u=a1{encoded}">Python &amp; docs</a></h2><p>Read&nbsp;the documentation</p></li>'
    assert engine._parse_bing(html) == [{"url": url, "title": "Python & docs", "snippet": "Read\xa0the documentation", "engine": "bing"}]


def test_ddg_accepts_reordered_attributes_and_decodes_redirect():
    html = '<div class="result"><a href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2F" class="result__a">Python &amp; docs</a><p class="result__snippet">Official <b>documentation</b></p></div>'
    hit = engine._parse_ddg(html)[0]
    assert hit["url"] == "https://docs.python.org/3/"
    assert hit["title"] == "Python & docs"
    assert hit["snippet"] == "Official documentation"


def test_failed_api_falls_back_to_keyless_results(monkeypatch, isolated_search):
    isolated_search.tavily_api_key = "test-key"
    called = []

    async def fetch(url, **kwargs):
        called.append((url, kwargs))
        if "tavily" in url:
            return fetched(status=401)
        return fetched('<li class="b_algo"><h2><a href="https://docs.python.org/">Python docs</a></h2><p>Official documentation</p></li>')

    monkeypatch.setattr(engine, "relai_fetch", fetch)
    result = asyncio.run(engine.relai_search("Python documentation"))
    assert result["hits"][0]["url"] == "https://docs.python.org/"
    assert result["status"] == "complete"
    assert result["attempted_providers"] == ["tavily", "bing"]
    assert result["errors"] == ["tavily: authentication or account credits required"]
    assert all(kwargs["retries"] == 0 and kwargs["cache"] is False for _, kwargs in called)
    assert "test-key" not in json.dumps(result)


def test_no_key_does_not_call_jina_and_failed_html_falls_back(monkeypatch):
    calls = []

    async def fetch(url, **kwargs):
        calls.append(url)
        if "bing.com" in url:
            return fetched(status=403, blocked=True)
        return fetched('<div class="result"><a class="result__a" href="https://docs.python.org/">Python docs</a></div>')

    monkeypatch.setattr(engine, "relai_fetch", fetch)
    result = asyncio.run(engine.relai_search("Python documentation"))
    assert result["engine"] == "duckduckgo"
    assert result["attempts"] == 2
    assert not any("jina" in url for url in calls)
    assert "blocked" in result["errors"][0]


def test_no_results_report_clear_failure_and_preserve_router_contract(monkeypatch):
    async def fetch(url, **kwargs):
        return fetched("<html>No matches for this query</html>")

    monkeypatch.setattr(engine, "relai_fetch", fetch)
    result = asyncio.run(engine.relai_search("missing-example-query"))
    assert result["hits"] == []
    assert result["tried"] == 0
    assert result["attempts"] == len(engine.ENGINES)
    assert result["status"] == "unavailable"
    assert result["blocked"] is True
    assert "No search provider" in result["message"]
    assert all("no usable results" in error for error in result["errors"])


def test_slow_provider_is_cancelled_and_fallback_can_succeed(monkeypatch, isolated_search):
    isolated_search.tavily_api_key = "test-key"
    monkeypatch.setattr(engine, "PROVIDER_TIMEOUT_SECONDS", 0.02)
    cancelled = []

    async def slow(*args):
        try:
            await asyncio.sleep(5)
        finally:
            cancelled.append(True)

    async def fetch(url, **kwargs):
        return fetched('<li class="b_algo"><h2><a href="https://docs.python.org/">Docs</a></h2></li>')

    monkeypatch.setattr(engine, "_tavily_search", slow)
    monkeypatch.setattr(engine, "relai_fetch", fetch)
    result = asyncio.run(engine.relai_search("documentation"))
    assert result["hits"]
    assert cancelled == [True]
    assert result["errors"] == ["tavily: timed out"]


def test_total_deadline_bounds_slow_fallbacks(monkeypatch):
    monkeypatch.setattr(engine, "SEARCH_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(engine, "PROVIDER_TIMEOUT_SECONDS", 1)

    async def fetch(*args, **kwargs):
        await asyncio.sleep(5)

    monkeypatch.setattr(engine, "relai_fetch", fetch)
    started = time.monotonic()
    result = asyncio.run(engine.relai_search("documentation"))
    assert time.monotonic() - started < 0.5
    assert result["attempted_providers"] == ["bing"]
    assert "time budget" in result["errors"][-1]


def test_invalid_urls_are_filtered_and_result_schema_is_normalized(monkeypatch, isolated_search):
    isolated_search.tavily_api_key = "test-key"

    async def provider(*args):
        return {"hits": [
            {"url": "javascript:alert(1)", "title": "Bad"},
            {"url": "http://127.0.0.1/private", "title": "Private"},
            {"url": "https://docs.python.org / tutorial", "title": "Citation"},
            {"url": "https://[invalid", "title": "Malformed"},
            {"url": "https://docs.python.org/", "title": "<b>Python &amp; docs</b>", "snippet": None},
            {"url": "https://docs.python.org/#duplicate", "title": "Duplicate"},
        ]}

    monkeypatch.setattr(engine, "_tavily_search", provider)
    result = asyncio.run(engine.relai_search("documentation"))
    assert result["hits"] == [{"url": "https://docs.python.org/", "title": "Python & docs", "snippet": "", "engine": "tavily"}]


def test_api_null_results_produce_a_clear_error(monkeypatch, isolated_search):
    isolated_search.jina_api_key = "test-key"

    async def fetch(*args, **kwargs):
        return fetched('{"data": null}')

    monkeypatch.setattr(engine, "relai_fetch", fetch)
    result = asyncio.run(engine._jina_search("documentation"))
    assert result == {"hits": [], "error": "provider returned no result list"}


def test_provider_exception_does_not_leak_credentials(monkeypatch, isolated_search):
    isolated_search.tavily_api_key = "test-key"
    monkeypatch.setattr(engine, "ENGINES", [])

    async def provider(*args):
        raise ValueError("test-key private endpoint")

    monkeypatch.setattr(engine, "_tavily_search", provider)
    result = asyncio.run(engine.relai_search("documentation"))
    assert result["errors"] == ["tavily: request failed (ValueError)"]
    assert "test-key" not in json.dumps(result)


def test_empty_query_does_not_contact_providers():
    with pytest.raises(ValueError, match="query"):
        asyncio.run(engine.relai_search("   "))
