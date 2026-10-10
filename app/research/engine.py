"""Research engine — multi-step research loop.

Runs: PLAN → SEARCH → OPEN → EXTRACT → ENOUGH? → synthesize

with hard, configurable budgets (searches, pages, runtime, depth) checked at
every step, and early stoppage once enough independent evidence exists. Pages
that trip the prompt-injection scan are quarantined.
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from ..core.types import (
    Evidence, SearchAccessMode, SearchResult, Source, SurfAction,
    SurfPlan, SurfSession, SurfSessionStatus,
)
from ..core.guard import assert_safe_url
from ..evidence.scoring import attach_scores, score_source, _objective_keywords, _score_relevance
from ..evidence.store import EvidenceStore
from ..search.router import route_search, run_deep_search
from ..extraction.page_data import extract_page_data
from ..core.http import relai_fetch

# ── In-memory session store ───────────────────────────────────────────

_sessions: dict[str, SurfSession] = {}
_stores: dict[str, EvidenceStore] = {}


def create_session(user_id: str, objective: str, plan: SurfPlan) -> SurfSession:
    session = SurfSession(
        id=uuid.uuid4().hex[:16],
        user_id=user_id,
        objective=objective,
        plan=plan,
    )
    _sessions[session.id] = session
    _stores[session.id] = EvidenceStore()
    return session


def get_session(session_id: str) -> Optional[SurfSession]:
    return _sessions.get(session_id)


def _get_store(session_id: str) -> Optional[EvidenceStore]:
    return _stores.get(session_id)


def record_action(session_id: str, kind: str, description: str) -> SurfAction:
    session = _sessions.get(session_id)
    if not session:
        return SurfAction(id="", kind=kind, description=description, status="error")
    action = SurfAction(
        id=uuid.uuid4().hex[:8],
        kind=kind,
        description=description,
        status="running",
    )
    session.actions.append(action)
    return action


def finish_action(action: SurfAction, status: str, detail: Optional[str] = None):
    action.status = status
    action.detail = detail
    action.ended_at = datetime.now(timezone.utc).isoformat()


# ── Plan Generation ───────────────────────────────────────────────────

async def plan_research(objective: str, depth: str = "standard") -> SurfPlan:
    """Generate a research plan with query variations."""
    keywords = _objective_keywords(objective)

    queries = [
        objective,
        f"{objective} overview",
        f"{objective} {datetime.now(timezone.utc).year}",
    ]
    if keywords:
        queries.append(" ".join(keywords[:5]))

    sources = ["web"]
    if any(w in objective.lower() for w in ["github", "code", "repo", "open source"]):
        sources.append("github")
    if any(w in objective.lower() for w in ["reddit", "community", "discussion"]):
        sources.append("reddit")

    return SurfPlan(
        objective=objective,
        queries=queries,
        sources=sources,
        required_information=[objective],
        verification_requirements=["cross-reference from multiple sources"],
        depth=depth,
    )


# ── Budget ────────────────────────────────────────────────────────────

def _resolve_limits(depth: str, overrides: Optional[dict] = None) -> dict:
    from ..core.types import DEPTH_LIMITS
    limits = DEPTH_LIMITS.get(depth, DEPTH_LIMITS["standard"]).copy()
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                limits[k] = v
    return limits


def _is_budget_exhausted(limits: dict, searches: int, pages: int, started_at: float) -> bool:
    if searches >= limits.get("max_searches", 8):
        return True
    if pages >= limits.get("max_pages", 12):
        return True
    if (time.time() - started_at) * 1000 >= limits.get("max_runtime_ms", 120_000):
        return True
    return False


# ── Evidence Extraction ───────────────────────────────────────────────

STOPWORDS = frozenset([
    "the", "and", "for", "with", "that", "this", "from", "have", "what", "when",
    "where", "about", "into", "over", "their", "they", "them", "there", "than",
    "then", "will", "would", "should", "could", "also", "were", "been", "being",
    "which", "while", "your", "our",
])


def _score_sentence(sentence: str, keywords: list[str], index: int) -> float:
    score = sum(1 for kw in keywords if kw in sentence.lower())
    if re.search(r"\d", sentence):
        score += 0.5
    if len(sentence) >= 120:
        score += 0.25
    if index < 4:
        score += 0.25
    return score


def extract_evidence(store: EvidenceStore, source: Source, text: str, objective: str):
    """Pull objective-relevant sentences from a page into the evidence store."""
    keywords = _objective_keywords(objective)
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    candidates = [
        (s.strip(), i) for i, s in enumerate(sentences)
        if 40 <= len(s.strip()) <= 320
    ]
    candidates.sort(key=lambda x: _score_sentence(x[0], keywords, x[1]), reverse=True)

    added = 0
    for sentence, _ in candidates[:8]:
        if added >= 4:
            break
        if store.count() >= 12:
            break
        store.add(Evidence(
            source_url=source.url,
            source_title=source.title,
            platform=None,
            claim=sentence[:200],
            excerpt=sentence,
            access_mode=SearchAccessMode.PUBLIC,
            confidence=source.score.overall if source.score else 0.6,
        ))
        added += 1


def _should_stop_early(store: EvidenceStore) -> bool:
    """Stop once we have decent multi-source evidence."""
    return store.count() >= 6 and store.distinct_sources() >= 3


# ── Hit Processing ────────────────────────────────────────────────────

async def process_hit(session_id: str, hit: SearchResult):
    """Fetch, scan and extract evidence from a single result URL."""
    store = _get_store(session_id)
    session = _sessions.get(session_id)
    if not store or not session:
        return

    source = Source(
        title=hit.title, url=hit.url, access_mode=hit.access_mode,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )
    score = score_source(source, session.objective)
    source.score = score
    store.track_source(source)

    try:
        url = assert_safe_url(hit.url)
    except ValueError:
        return

    action = record_action(session_id, "open", f"Opening: {hit.title[:60]}")

    try:
        result = await relai_fetch(url, timeout_ms=15_000, retries=2)
        if not result.ok:
            finish_action(action, "error", "Page could not be loaded")
            return

        is_json = "json" in result.content_type
        page = extract_page_data(
            html=None if is_json else result.text,
            json_str=result.text if is_json else None,
            url=result.url,
            status=result.status,
            content_type=result.content_type,
            max_chars=20_000,
            fetched_at=result.fetched_at,
        )

        finish_action(action, "done", f"{len(page.text):,} chars extracted")

        # Quarantine injection-suspect pages
        if page.injection and page.injection.suspect:
            record_action(session_id, "extract", f"Quarantined: {hit.title[:60]} (possible prompt injection)")
            return

        extract_evidence(store, source, page.text, session.objective)
    except Exception as e:
        finish_action(action, "error", str(e)[:200])


# ── Research Loop ─────────────────────────────────────────────────────

async def _run_research_loop(session_id: str, opts: dict):
    session = _sessions.get(session_id)
    if not session or not session.plan:
        if session:
            session.status = SurfSessionStatus.FAILED
            session.error = "Session or plan missing at loop start"
        return

    plan = session.plan
    limits = _resolve_limits(plan.depth, opts)
    searches = 0
    pages = 0
    started_at = time.time()
    deadline = asyncio.get_running_loop().time() + limits["max_runtime_ms"] / 1000

    try:
        queries = list(plan.queries)
        qi = 0

        while qi < len(queries):
            if _is_budget_exhausted(limits, searches, pages, started_at):
                break
            if _should_stop_early(_get_store(session_id)):
                break

            query = queries[qi]
            qi += 1
            searches += 1

            action = record_action(session_id, "search", f'Searching: "{query[:80]}"')

            try:
                async with asyncio.timeout_at(deadline):
                    if plan.depth in ("deep", "exhaustive"):
                        result = await run_deep_search(query)
                        hits = [SearchResult.model_validate({
                            **hit, "provider": hit.get("provider", hit.get("engine", "unknown")),
                        }) for hit in result["results"]]
                        provider, note = result["provider"], ""
                    else:
                        result = await route_search(query, intent="general")
                        hits, provider, note = result.results, result.provider, result.note or ""
            except TimeoutError:
                finish_action(action, "error", "Research time budget reached")
                break
            except Exception as e:
                finish_action(action, "error", str(e)[:200])
                continue

            if not hits:
                finish_action(action, "error", note or "No results")
                continue

            finish_action(action, "done", f"{len(hits)} results via {provider}")

            for hit in hits:
                if _is_budget_exhausted(limits, 0, pages, started_at):
                    break
                if _should_stop_early(_get_store(session_id)):
                    break
                pages += 1
                try:
                    async with asyncio.timeout_at(deadline):
                        await process_hit(session_id, hit)
                except TimeoutError:
                    for pending in session.actions:
                        if pending.status == "running":
                            finish_action(pending, "error", "Research time budget reached")
                    break

            # Re-query when evidence is thin
            if qi >= len(queries) and not _should_stop_early(_get_store(session_id)):
                store = _get_store(session_id)
                if store and store.count() < 3 and searches < limits["max_searches"]:
                    follow_up = f"{plan.objective} site:reddit.com OR site:github.com"
                    if follow_up not in queries:
                        queries.append(follow_up)

        sync_evidence(session_id)
        session.status = SurfSessionStatus.COMPLETE
        session.completed_at = datetime.now(timezone.utc).isoformat()

    except Exception as e:
        sync_evidence(session_id)
        session.status = SurfSessionStatus.FAILED
        session.error = str(e)[:500]
        session.completed_at = datetime.now(timezone.utc).isoformat()


def sync_evidence(session_id: str):
    """Sync evidence store to session."""
    store = _get_store(session_id)
    session = _sessions.get(session_id)
    if store and session:
        session.evidence = store.all()
        session.sources = [
            Source(
                title=s.title, url=s.url, access_mode="public",
                retrieved_at=s.retrieved_at, score=s.score,
            )
            for s in store.sources()
        ]


# ── Public API ────────────────────────────────────────────────────────

async def start_research(
    objective: str,
    user_id: str = "anonymous",
    depth: str = "standard",
    workspace_id: Optional[str] = None,
    max_searches: Optional[int] = None,
    max_pages: Optional[int] = None,
    max_runtime_ms: Optional[int] = None,
) -> SurfSession:
    """Start a research run. Returns session immediately; loop runs in background."""
    if not objective.strip():
        raise ValueError("A research objective is required")

    plan = await plan_research(objective, depth)
    session = create_session(user_id, objective.strip(), plan)
    if workspace_id:
        session.workspace_id = workspace_id

    # Run in background
    opts = {}
    if max_searches is not None:
        opts["max_searches"] = max_searches
    if max_pages is not None:
        opts["max_pages"] = max_pages
    if max_runtime_ms is not None:
        opts["max_runtime_ms"] = max_runtime_ms

    asyncio.create_task(_run_research_loop(session.id, opts))
    return session


def get_research(session_id: str) -> Optional[SurfSession]:
    """Get a live snapshot of a research session."""
    session = _sessions.get(session_id)
    if session:
        sync_evidence(session_id)
    return session
