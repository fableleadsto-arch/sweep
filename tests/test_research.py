import asyncio

import pytest

from app.core.types import SearchResult, SearchRunResult, SurfSessionStatus
from app.research import engine


@pytest.mark.parametrize("depth", ["standard", "deep"])
def test_research_uses_typed_hits_and_processes_final_search(monkeypatch, depth):
    processed = []

    async def standard(query, **kwargs):
        return SearchRunResult(provider="test", query=query, results=[
            SearchResult(url="https://example.com", title="Example", provider="test")])

    async def deep(query):
        return {"provider": "test", "results": [
            {"url": "https://example.com", "title": "Example", "engine": "test"}]}

    async def process(session_id, hit):
        assert isinstance(hit, SearchResult)
        processed.append(hit.url)

    async def run():
        plan = await engine.plan_research("test", depth)
        session = engine.create_session("test", "test", plan)
        try:
            await engine._run_research_loop(session.id, {"max_searches": 1})
            assert session.status == SurfSessionStatus.COMPLETE
            assert processed == ["https://example.com"]
        finally:
            engine._sessions.pop(session.id)
            engine._stores.pop(session.id)

    monkeypatch.setattr(engine, "route_search", standard)
    monkeypatch.setattr(engine, "run_deep_search", deep)
    monkeypatch.setattr(engine, "process_hit", process)
    asyncio.run(run())


def test_research_timeout_cancels_pending_search(monkeypatch):
    cancelled = []

    async def search(*args, **kwargs):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)

    async def run():
        session = engine.create_session("test", "test", await engine.plan_research("test"))
        try:
            await asyncio.wait_for(engine._run_research_loop(session.id, {"max_runtime_ms": 25}), 1)
            assert cancelled and session.completed_at
            assert all(action.status != "running" for action in session.actions)
        finally:
            engine._sessions.pop(session.id)
            engine._stores.pop(session.id)

    monkeypatch.setattr(engine, "route_search", search)
    asyncio.run(run())
