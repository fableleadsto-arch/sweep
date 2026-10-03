"""API routes — REST endpoints for the web intelligence platform.

Endpoints:
  GET  /health                           → service health
  POST /api/search                       → web search (with optional platform routing)
  POST /api/research                     → start research run
  GET  /api/research/{id}                → get research session status
  POST /api/browse                       → create browse session
  POST /api/browse/{id}/navigate         → navigate in browse session
  GET  /api/browse/{id}/page             → current page of session
  POST /api/extract                      → extract structured data from URL
  GET  /api/platforms                     → list platform adapters
  GET  /api/providers                    → search provider health/status
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..core.types import (
    NavigationCommand, SearchOptions, SurfSession,
)
from ..search.engine import relai_search
from ..search.router import route_search, run_deep_search
from ..research.engine import start_research, get_research
from ..browser.sessions import (
    create_browse_session, navigate, current_page, close_browse_session,
    session_count,
)
from ..extraction.page_data import extract_page_data
from ..core.http import relai_fetch

router = APIRouter(prefix="/api")


# ── Request/Response Models ───────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    limit: int = Field(default=10, ge=1, le=50)
    site: Optional[str] = None
    time_range: Optional[str] = None
    intent: str = "general"
    aggregate: bool = False
    platform: Optional[str] = None  # "youtube", "x", "linkedin", "instagram", "reddit", "github"


class ResearchRequest(BaseModel):
    objective: str = Field(min_length=1, max_length=4000)
    depth: str = "standard"
    user_id: str = "anonymous"
    workspace_id: Optional[str] = None
    max_searches: Optional[int] = Field(default=None, ge=1, le=30)
    max_pages: Optional[int] = Field(default=None, ge=1, le=100)
    max_runtime_ms: Optional[int] = Field(default=None, ge=1000, le=300000)


class NavigateRequest(BaseModel):
    kind: str = "goto"
    target: Optional[str] = None
    value: Optional[str] = None
    selector: Optional[str] = None
    direction: Optional[str] = None


class ExtractRequest(BaseModel):
    url: str = Field(min_length=1, max_length=8192)
    max_chars: int = Field(default=12_000, ge=500, le=60000)


# ── Search ────────────────────────────────────────────────────────────

@router.post("/search")
async def api_search(req: SearchRequest):
    """Web search with provider routing.

    When a platform is specified, routes to the platform adapter's native
    search instead of the generic web search pipeline.
    """
    # Platform-specific search
    if req.platform:
        from ..platforms import get_adapter_by_platform
        adapter = get_adapter_by_platform(req.platform)
        if adapter:
            hits, note, access = await adapter.search(req.query, limit=req.limit)
            return {
                "provider": f"platform_{req.platform}",
                "query": req.query,
                "results": [h.model_dump() for h in hits],
                "blocked": access.value == "unavailable",
                "note": note,
                "errors": [],
            }

    # Generic web search
    opts = SearchOptions(
        limit=req.limit,
        site=req.site,
        time_range=req.time_range,
        aggregate=req.aggregate,
    )
    result = await route_search(req.query, intent=req.intent, options=opts)
    return {
        "provider": result.provider,
        "query": result.query,
        "results": [r if isinstance(r, dict) else r.model_dump() for r in result.results],
        "blocked": result.blocked,
        "note": result.note,
        "errors": result.errors,
    }


# ── Research ──────────────────────────────────────────────────────────

@router.post("/research")
async def api_research_start(req: ResearchRequest):
    """Start a research run."""
    try:
        session = await start_research(
            objective=req.objective,
            user_id=req.user_id,
            depth=req.depth,
            workspace_id=req.workspace_id,
            max_searches=req.max_searches,
            max_pages=req.max_pages,
            max_runtime_ms=req.max_runtime_ms,
        )
        return session.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/research/{session_id}")
async def api_research_get(session_id: str):
    """Get research session status."""
    session = get_research(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.model_dump()


# ── Browse ────────────────────────────────────────────────────────────

@router.post("/browse")
async def api_browse_create():
    """Create a new browse session."""
    session = create_browse_session()
    return session


@router.post("/browse/{session_id}/navigate")
async def api_browse_navigate(session_id: str, req: NavigateRequest):
    """Navigate in a browse session."""
    cmd = NavigationCommand(
        kind=req.kind,
        target=req.target,
        value=req.value,
        selector=req.selector,
        direction=req.direction,
    )
    result = await navigate(session_id, cmd)
    return result.model_dump()


@router.get("/browse/{session_id}/page")
async def api_browse_page(session_id: str):
    """Get current page of a browse session."""
    page = current_page(session_id)
    if not page:
        return {"error": "No page loaded"}
    return page.model_dump()


@router.delete("/browse/{session_id}")
async def api_browse_close(session_id: str):
    """Close a browse session."""
    close_browse_session(session_id)
    return {"ok": True}


# ── Extraction ────────────────────────────────────────────────────────

@router.get("/platforms")
async def api_platforms():
    """List registered platform adapters."""
    from ..platforms import list_adapters
    return {"adapters": list_adapters()}


@router.post("/extract")
async def api_extract(req: ExtractRequest):
    """Extract structured data from a URL.

    Uses platform-specific adapters when the URL matches a known platform
    (YouTube, X, LinkedIn, Instagram, Reddit, GitHub), otherwise falls back
    to generic HTML extraction.
    """
    try:
        # Try platform-specific extraction first
        from ..platforms import get_adapter_for_url
        adapter = get_adapter_for_url(req.url)
        if adapter:
            page = await adapter.extract_page(req.url)
            if page:
                return page.model_dump()

        # Generic extraction fallback
        result = await relai_fetch(req.url, timeout_ms=15_000, retries=2)
        if not result.ok:
            raise HTTPException(status_code=502, detail=result.error or "Failed to fetch URL")

        is_json = "json" in result.content_type
        page = extract_page_data(
            html=None if is_json else result.text,
            json_str=result.text if is_json else None,
            url=result.url,
            status=result.status,
            content_type=result.content_type,
            max_chars=req.max_chars,
            fetched_at=result.fetched_at,
        )
        return page.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:500])


# ── Provider Health ───────────────────────────────────────────────────

@router.get("/providers")
async def api_providers():
    """List search provider status."""
    from ..config import get_settings
    settings = get_settings()
    return {
        "providers": [
            {"name": "keyless", "available": True, "capabilities": ["site_filter"]},
            {"name": "tavily", "available": bool(settings.tavily_api_key), "capabilities": ["site_filter", "news"]},
            {"name": "exa", "available": bool(settings.exa_api_key), "capabilities": ["site_filter", "technical", "news"]},
            {"name": "searxng", "available": bool(settings.searxng_base_url), "capabilities": ["meta_search"]},
            {"name": "jina", "available": True, "capabilities": ["free_tier"]},
        ],
        "configured": settings.search_providers_configured,
    }


# ── ML / Compute ─────────────────────────────────────────────────────

class ComputeRequest(BaseModel):
    task: str
    capability: Optional[str] = None
    data: Optional[object] = None
    params: Optional[dict] = None


@router.get("/compute/capabilities")
async def api_compute_capabilities():
    """List available ML/compute capabilities."""
    try:
        from companion.capabilities import list_capabilities
        caps = list_capabilities()
        return [c.model_dump() for c in caps]
    except Exception as e:
        return {"error": str(e), "capabilities": []}


@router.post("/compute/run")
async def api_compute_run(req: ComputeRequest):
    """Run a task through the best-fit ML framework."""
    try:
        from companion.capabilities import CapabilityEngine, resolve_capability
        cap, score = resolve_capability(req.task, req.capability or "", req.data)
        if not cap.available:
            return {
                "ok": False,
                "capability": cap.id,
                "error": f"{cap.label} needs: {', '.join(cap.missing_libraries)}",
            }
        import asyncio
        payload = {
            "task": req.task,
            "capability": cap.id,
            "data": req.data,
            "params": req.params or {},
        }
        outcome = await asyncio.to_thread(cap.tool, payload)
        return {
            "ok": bool(outcome.get("result")),
            "capability": cap.id,
            "result": outcome.get("result"),
            "summary": outcome.get("summary", "Done."),
            "libraries_used": outcome.get("libraries_used", []),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)[:500]}


@router.get("/compute/torch")
async def api_torch_info():
    """PyTorch status and info."""
    try:
        import torch
        return {
            "available": True,
            "version": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
            "devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    except ImportError:
        return {"available": False, "error": "PyTorch not installed"}


# ── Health ────────────────────────────────────────────────────────────

@router.get("/health")
async def api_health():
    """Service health check."""
    from ..config import get_settings
    settings = get_settings()
    torch_ok = False
    try:
        import torch
        torch_ok = True
    except ImportError:
        pass
    return {
        "status": "ok",
        "service": "Sweep API",
        "version": "2.0.0",
        "python": True,
        "pytorch": torch_ok,
        "playwright": settings.playwright_configured,
        "search_providers": settings.search_providers_configured,
        "browse_sessions": session_count(),
    }
