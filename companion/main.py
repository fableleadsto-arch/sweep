"""FastAPI application — the RelayAI brain service HTTP layer.

Endpoints (OpenAPI docs at `/docs`):

    GET  /health                      → service + provider health
    POST /api/companion/turn          → one companion turn (legacy-compatible)
    POST /api/brain/turn              → alias of the companion turn
    POST /api/brain/overview          → daily overview briefing
    POST /api/brain/context           → live relay workspace context bundle
    POST /api/brain/rag               → RAG knowledge retrieval
    POST /api/brain/memory/search     → memory lookup
    POST /api/brain/memory/remember   → persist a memory
    GET  /api/brain/memory/summary    → memory stats for a user
    POST /api/brain/embed             → embedding pipeline
    GET  /api/brain/capabilities      → the computational-toolbox catalog
    POST /api/brain/compute           → run a task through the right framework
    GET  /api/brain/compute/backends      → compute backend status (Settings → Compute)
    POST /api/brain/compute/backends/{id}/toggle   → enable/disable a backend
    POST /api/brain/compute/backends/{id}/install  → controlled wheel install
    GET  /api/brain/compute/diagnostics → full hardware/framework report
    POST /api/brain/compute/schedule    → dry-run hardware-aware scheduling
    GET  /api/brain/native/models       → native model registry (real params)
    GET  /api/brain/native/models/{name}→ one native model record
    GET  /api/brain/native/recommend    → hardware-aware scale recommendation
    POST /api/brain/native/generate     → generate with a trained native model
    POST /api/brain/native/route        → native-vs-external routing decision
    POST /api/brain/execute           → run a generated Python script (sandboxed)
    POST /api/brain/plan              → decompose a request into tool steps
    POST /api/brain/plan/chat         → deterministic request routing (no LLM)
    POST /api/brain/agent/turn        → full agent loop
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .neural.registry import ModelRegistry

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger("companion")

from . import __version__


def log_vendor_inventory() -> None:
    """Log what framework source + bundled wheels are stored locally.

    Cheap (no imports of heavy frameworks) and safe on any deployment: the
    vendor root may be absent in stripped images, in which case it logs the
    gap instead of failing.
    """
    try:
        from .tools.vendor_loader import inventory_summary

        inv = inventory_summary()
    except (ImportError, OSError) as exc:  # boot must never fail on inventory
        logger.warning("vendor inventory unavailable: %s", exc)
        return
    logger.info(
        "vendor: %d importable pkg(s) %s | source archives %d/%d | wheels %d/%d | root=%s",
        len(inv["importable_packages"]),
        ",".join(inv["importable_packages"][:5]) + ("..." if len(inv["importable_packages"]) > 5 else ""),
        inv["source_archives_present"],
        inv["source_archives_total"],
        inv["wheels_present"],
        inv["wheels_total"],
        inv["root"],
    )
    if inv["wheels_missing"]:
        missing = ", ".join(f"{m['name']} ({m['platform']})" for m in inv["wheels_missing"])
        logger.info("vendor: %d bundled wheel(s) not on disk — %s", len(inv["wheels_missing"]), missing)
from .brain import CompanionBrain
from .capabilities import CapabilityEngine
from .compute.backend_manager import BackendManager
from .config import BrainSettings, get_settings
from .embeddings import embed_batch
from .execution import execute_python
from .providers import ProviderChain
from app.core.auth import authorize
from .orchestrator import (
    AgentTurnRequest,
    AgentTurnResponse,
    BrainPlanRequest,
    BrainPlanResponse,
    Orchestrator,
    OrchestratorModel,
    ProviderChainModel,
)
from .planning import plan_chat_request
from .rag import RagService
from .relaydata import RelayContextService
from .schemas import (
    CapabilityInfo,
    ComputeBackendInstallRequest,
    ComputeBackendStatus,
    ComputeBackendToggleRequest,
    ComputeDiagnosticsResponse,
    ComputeInstallResult,
    ComputeRequest,
    ComputeResult,
    ComputeScheduleRequest,
    ComputeScheduleResponse,
    ContextRequest,
    ContextResponse,
    EmbedRequest,
    EmbedResult,
    ErrorResponse,
    ExecuteRequest,
    ExecuteResult,
    HealthResponse,
    MemoryEntry,
    MemoryRememberRequest,
    MemorySearchRequest,
    NativeGenerateRequest,
    NativeGenerateResult,
    NativeModelRecord,
    NativeRecommendResponse,
    NativeRouteRequest,
    NativeRouteResult,
    OverviewRequest,
    OverviewResponse,
    ProviderHealth,
    RagRequest,
    RagResult,
    TurnRequest,
    TurnResponse,
)
from .contracts import ChatPlanRequest, ChatPlanResponse

@asynccontextmanager
async def lifespan(_: FastAPI):
    """App lifecycle — boots the vendor inventory + background ingestion loop."""
    log_vendor_inventory()
    try:
        from .ingest.scheduler import start_background_scheduler, stop_background_scheduler

        scheduler = await start_background_scheduler()
    except Exception:  # noqa: BLE001 - a scheduler failure must never block boot
        logger.exception("failed to start ingestion scheduler")
        scheduler = None
    try:
        yield
    finally:
        if scheduler is not None:
            try:
                await stop_background_scheduler()
            except Exception:  # noqa: BLE001
                logger.exception("failed to stop ingestion scheduler")


app = FastAPI(
    title="RelayAI Brain Service",
    description="Python companion brain — LLM fallback, memory, RAG retrieval.",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[s.strip() for s in get_settings().cors_origins.split(",") if s.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_token(
    authorization: str | None = Header(default=None),
    settings: BrainSettings = Depends(get_settings),
) -> None:
    """Required bearer-token gate for /api/* routes.

    Every API call must present `Authorization: Bearer <token>`.
    Missing configuration fails closed with 503; bad credentials return 401.
    Health, `/docs` and `/` stay open so uptime probes and the OpenAPI UI
    work without a token.
    """
    authorize(authorization, settings.brain_service_token)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Turn unhandled errors into structured JSON instead of a bare 500 text body.

    Callers (the classic Node API proxy, direct clients) can then reliably parse
    the failure. The traceback is still logged server-side for debugging.
    """
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


def brain(settings: BrainSettings) -> CompanionBrain:
    return CompanionBrain(settings)


def get_orchestrator(settings: BrainSettings = Depends(get_settings)) -> Orchestrator:
    """The agentic-loop orchestrator (override in tests)."""
    return Orchestrator()


def get_orchestrator_model(
    settings: BrainSettings = Depends(get_settings),
) -> OrchestratorModel:
    """The default model adapter — the companion provider chain (override in tests)."""
    return ProviderChainModel(ProviderChain(settings))


def rag(settings: BrainSettings) -> RagService:
    return RagService(settings)


def relay(settings: BrainSettings) -> RelayContextService:
    return RelayContextService(settings)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health(
    settings: BrainSettings = Depends(get_settings),
) -> HealthResponse:
    health_map = brain(settings).providers.health()
    return HealthResponse(
        status="ok",
        model=settings.gemini_model,
        providers=ProviderHealth(
            **{k: health_map.get(k, False) for k in ProviderHealth.model_fields}
        ),
        python=sys.version.split()[0],
        version=__version__,
    )


@app.post(
    "/api/companion/turn",
    response_model=TurnResponse,
    responses={500: {"model": ErrorResponse}},
    tags=["brain"],
    dependencies=[Depends(require_token)],
)
@app.post(
    "/api/brain/turn",
    response_model=TurnResponse,
    responses={500: {"model": ErrorResponse}},
    tags=["brain"],
    include_in_schema=False,
    dependencies=[Depends(require_token)],
)
async def companion_turn(
    request: TurnRequest,
    settings: BrainSettings = Depends(get_settings),
) -> TurnResponse:
    """Complete one turn of the companion conversation loop."""
    return await brain(settings).process_turn(
        user_id=request.user_id,
        workspace_id=request.workspace_id,
        message=request.message,
        history=request.history,
    )


@app.post(
    "/api/brain/rag", response_model=RagResult, tags=["brain"], dependencies=[Depends(require_token)]
)
async def rag_retrieve(
    request: RagRequest,
    settings: BrainSettings = Depends(get_settings),
) -> RagResult:
    """Retrieve relevant knowledge-base context for a query."""
    return await rag(settings).retrieve(
        request.workspace_id,
        request.query,
        max_chunks=request.max_chunks,
        min_score=request.min_score,
    )


@app.post(
    "/api/brain/overview",
    response_model=OverviewResponse,
    tags=["relay"],
    dependencies=[Depends(require_token)],
)
async def brain_overview(
    request: OverviewRequest,
    settings: BrainSettings = Depends(get_settings),
) -> OverviewResponse:
    """Assemble the daily overview briefing for a user's workspace."""
    briefing = await relay(settings).overview(
        request.user_id,
        request.workspace_id,
        briefing_type=request.briefing_type,
    )
    return OverviewResponse(briefing=briefing)


@app.post(
    "/api/brain/context",
    response_model=ContextResponse,
    tags=["relay"],
    dependencies=[Depends(require_token)],
)
async def brain_context(
    request: ContextRequest,
    settings: BrainSettings = Depends(get_settings),
) -> ContextResponse:
    """Gather the live relay workspace context a brain turn would use."""
    bundle = await relay(settings).build_context(
        request.user_id,
        request.workspace_id,
        request.message,
    )
    return ContextResponse(
        profile={
            "name": bundle.profile.greeting_name,
            "timezone": bundle.profile.timezone,
            "communication_style": bundle.profile.communication_style,
            "preferred_channel": bundle.profile.preferred_channel,
            "response_length": bundle.profile.response_length,
            "mood_trend": bundle.profile.mood_trend,
        },
        overview_requested=bundle.overview_requested,
        overview=bundle.overview,
        workspace={
            "workspace_id": bundle.workspace.workspace_id,
            "workspace_name": bundle.workspace.workspace_name,
            "pending_approvals": bundle.workspace.pending_approvals,
            "unread_conversations": bundle.workspace.unread_conversations,
            "open_conversations": bundle.workspace.open_conversations,
            "active_workflows": bundle.workspace.active_workflows,
            "workflow_runs_24h": bundle.workspace.workflow_runs_24h,
            "agent_runs_24h": bundle.workspace.agent_runs_24h,
            "leads_count": bundle.workspace.leads_count,
            "contacts_count": bundle.workspace.contacts_count,
        },
        graph_facts=bundle.graph_facts,
        companion_tasks=bundle.companion_tasks,
    )


@app.post(
    "/api/brain/memory/search",
    response_model=list[MemoryEntry],
    tags=["memory"],
    dependencies=[Depends(require_token)],
)
async def memory_search(
    request: MemorySearchRequest,
    settings: BrainSettings = Depends(get_settings),
) -> list[MemoryEntry]:
    """Search a user's persistent memory."""
    return await brain(settings).memory.search(
        request.user_id,
        request.query,
        workspace_id=request.workspace_id,
        limit=request.limit,
    )


@app.post(
    "/api/brain/memory/remember",
    response_model=MemoryEntry,
    tags=["memory"],
    dependencies=[Depends(require_token)],
)
async def memory_remember(
    request: MemoryRememberRequest,
    settings: BrainSettings = Depends(get_settings),
) -> MemoryEntry:
    """Persist a memory for a user."""
    return await brain(settings).memory.remember(
        request.user_id,
        request.content,
        workspace_id=request.workspace_id,
        kind=request.kind,
        tags=request.tags,
        source=request.source,
        confidence=request.confidence,
        expires_at=request.expires_at,
    )


@app.get(
    "/api/brain/memory/summary",
    response_model=dict[str, Any],
    tags=["memory"],
    dependencies=[Depends(require_token)],
)
async def memory_summary(
    user_id: str,
    settings: BrainSettings = Depends(get_settings),
) -> dict[str, Any]:
    """Summary of memory usage for a user."""
    return brain(settings).memory.summary(user_id)


@app.post(
    "/api/brain/embed",
    response_model=EmbedResult,
    tags=["brain"],
    dependencies=[Depends(require_token)],
)
async def embed_texts(
    request: EmbedRequest,
    settings: BrainSettings = Depends(get_settings),
) -> EmbedResult:
    """Embed a batch of texts through the shared embedding pipeline."""
    embeddings = await embed_batch(request.texts, settings)
    return EmbedResult(embeddings=embeddings)


def compute_engine(settings: BrainSettings) -> CapabilityEngine:
    return CapabilityEngine(settings)


@app.get(
    "/api/brain/capabilities",
    response_model=list[CapabilityInfo],
    tags=["compute"],
    dependencies=[Depends(require_token)],
)
async def brain_capabilities(
    settings: BrainSettings = Depends(get_settings),
) -> list[CapabilityInfo]:
    """List every framework in Relay's computational toolbox + availability.

    Lets callers (and the brain itself) see exactly which capabilities are
    installed and ready without loading any of them.
    """
    return compute_engine(settings).catalog()


@app.post(
    "/api/brain/compute",
    response_model=ComputeResult,
    tags=["compute"],
    dependencies=[Depends(require_token)],
)
async def brain_compute(
    request: ComputeRequest,
    settings: BrainSettings = Depends(get_settings),
) -> ComputeResult:
    """Run a task through the best-fit framework (auto-selected or explicit).

    The engine maps the task to a capability (Pandas/NumPy for a CSV, OpenCV
    for an image, SymPy for an equation, Scikit-learn for a model, ...),
    lazily loads only that framework, executes, and returns a structured
    result. Heavy frameworks are never loaded for unrelated requests.
    """
    return await compute_engine(settings).run(request)


@app.post(
    "/api/brain/execute",
    response_model=ExecuteResult,
    tags=["compute"],
    dependencies=[Depends(require_token)],
)
async def brain_execute(
    request: ExecuteRequest,
    settings: BrainSettings = Depends(get_settings),
) -> ExecuteResult:
    """Run trusted Python in a filtered child process when explicitly enabled.

    AST checks are defense in depth, not an operating-system security boundary.
    Resource limits vary by platform; arbitrary untrusted code needs a VM/container.
    """
    if not settings.enable_compute or not settings.allow_code_execution:
        return ExecuteResult(
            ok=False,
            error="Generated-code execution is disabled. Set ALLOW_CODE_EXECUTION=true only for trusted scripts.",
        )
    outcome = await execute_python(request.code, request.env, request.timeout_ms)
    return ExecuteResult(**outcome)


def compute_manager(settings: BrainSettings) -> BackendManager:
    return BackendManager(settings)


@app.get(
    "/api/brain/compute/backends",
    response_model=list[ComputeBackendStatus],
    tags=["compute"],
    dependencies=[Depends(require_token)],
)
async def brain_compute_backends(
    settings: BrainSettings = Depends(get_settings),
) -> list[ComputeBackendStatus]:
    """List every compute backend + its live availability/enable state.

    Import-free probes: heavy frameworks are never loaded by this endpoint.
    """
    statuses = compute_manager(settings).status()
    return [ComputeBackendStatus(**s.__dict__) for s in statuses]


@app.post(
    "/api/brain/compute/backends/{backend_id}/toggle",
    response_model=ComputeBackendStatus,
    tags=["compute"],
    dependencies=[Depends(require_token)],
)
async def brain_compute_toggle(
    backend_id: str,
    request: ComputeBackendToggleRequest,
    settings: BrainSettings = Depends(get_settings),
) -> ComputeBackendStatus:
    """Enable/disable a compute backend (persisted to compute_config_file)."""
    manager = compute_manager(settings)
    if not manager.set_enabled(backend_id, request.enabled):
        raise HTTPException(status_code=404, detail=f"Unknown backend '{backend_id}'.")
    status = manager.status_of(backend_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Unknown backend '{backend_id}'.")
    return ComputeBackendStatus(**status.__dict__)


@app.post(
    "/api/brain/compute/backends/{backend_id}/install",
    response_model=ComputeInstallResult,
    tags=["compute"],
    dependencies=[Depends(require_token)],
)
async def brain_compute_install(
    backend_id: str,
    request: ComputeBackendInstallRequest,
    settings: BrainSettings = Depends(get_settings),
) -> ComputeInstallResult:
    """Install a backend through the controlled wheel-install path.

    dry_run defaults to True — validates against the local wheel registry
    without touching the environment. A real install additionally requires
    COMPANION_ALLOW_WHEEL_INSTALL=1.
    """
    if not settings.enable_compute:
        return ComputeInstallResult(
            ok=False, backend=backend_id, error="Compute is disabled (ENABLE_COMPUTE=false)."
        )
    result = compute_manager(settings).install(backend_id, dry_run=request.dry_run)
    return ComputeInstallResult(
        ok=bool(result.get("ok")),
        backend=backend_id,
        method=result.get("method") or "",
        dry_run=bool(result.get("dry_run", True)),
        wheel=result.get("wheel") or "",
        profile=result.get("profile") or "",
        already_available=bool(result.get("already_available")),
        error=result.get("error") or "",
        summary=result.get("summary") or result.get("note") or "",
    )


@app.get(
    "/api/brain/compute/diagnostics",
    response_model=ComputeDiagnosticsResponse,
    tags=["compute"],
    dependencies=[Depends(require_token)],
)
async def brain_compute_diagnostics(
    settings: BrainSettings = Depends(get_settings),
    include_heavy: bool = False,
) -> ComputeDiagnosticsResponse:
    """Full compute-environment diagnostics (hardware, frameworks, backends).

    Runs fast smoke tests on enabled backends so the report verifies each
    backend actually executes, not just that it imports. Pass
    ``include_heavy=true`` to also smoke-test slow-import frameworks
    (TensorFlow).
    """
    from .compute.diagnostics import diagnose

    return ComputeDiagnosticsResponse(**diagnose(settings, include_heavy=include_heavy).to_dict())


@app.post(
    "/api/brain/compute/schedule",
    response_model=ComputeScheduleResponse,
    tags=["compute"],
    dependencies=[Depends(require_token)],
)
async def brain_compute_schedule(
    request: ComputeScheduleRequest,
    settings: BrainSettings = Depends(get_settings),
) -> ComputeScheduleResponse:
    """Dry-run: choose the best backend for a task (no execution).

    Lets the UI and the orchestrator preview hardware-aware scheduling and its
    reasoning before running anything.
    """
    from .compute.base import ComputeTask, ComputeTaskKind
    from .compute.scheduler import schedule as schedule_task

    task = ComputeTask(
        kind=ComputeTaskKind.CAPABILITY,
        capability=request.capability,
        payload={
            "task": request.task,
            "capability": request.capability,
            "data": request.data,
            "params": request.params or {},
            "image_base64": request.image_base64,
            "_settings": settings,
        },
        model_format=request.model_format,
        framework_hint=request.framework,
        device_preference=request.device_preference,
        estimated_memory_mb=request.estimated_memory_mb,
        precision=request.precision,
    )
    decision = schedule_task(task, manager=compute_manager(settings))
    return ComputeScheduleResponse(
        task=request.task,
        capability=request.capability,
        backend=decision.backend,
        device=decision.device,
        score=decision.score,
        reason=decision.reason,
        candidates=decision.candidates,
        rejected=decision.rejected,
        ok=decision.backend is not None,
    )


@app.post(
    "/api/brain/plan",
    response_model=BrainPlanResponse,
    tags=["orchestrator"],
    dependencies=[Depends(require_token)],
)
async def brain_plan(
    request: BrainPlanRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    model: OrchestratorModel = Depends(get_orchestrator_model),
) -> BrainPlanResponse:
    """Decompose a request into an ordered tool plan (no execution)."""
    return await orchestrator.plan(request, model)


# ── Relay Native neural stack ──────────────────────────────────────────


def native_registry() -> ModelRegistry:
    from .neural.registry import DEFAULT_REGISTRY_DIR, ModelRegistry

    return ModelRegistry(DEFAULT_REGISTRY_DIR)


@app.get(
    "/api/brain/native/models",
    response_model=list[NativeModelRecord],
    tags=["native"],
    dependencies=[Depends(require_token)],
)
async def brain_native_models() -> list[NativeModelRecord]:
    """List registered native models with real (computed) parameter counts."""
    try:
        models = native_registry().list_models()
    except Exception as exc:  # noqa: BLE001 - registry must never break the API
        raise HTTPException(status_code=500, detail=f"Native registry unavailable: {exc}")
    return [NativeModelRecord(**m.to_dict()) for m in models]


@app.get(
    "/api/brain/native/models/{name}",
    response_model=NativeModelRecord,
    tags=["native"],
    dependencies=[Depends(require_token)],
)
async def brain_native_model(name: str) -> NativeModelRecord:
    """Details for a single native model (real parameter count included)."""
    try:
        rec = native_registry().record(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return NativeModelRecord(**rec.to_dict())


@app.get(
    "/api/brain/native/recommend",
    response_model=NativeRecommendResponse,
    tags=["native"],
    dependencies=[Depends(require_token)],
)
async def brain_native_recommend() -> NativeRecommendResponse:
    """Hardware-aware recommendation of the largest scale that truly fits."""
    from .neural.selection import recommend_model

    fit = recommend_model()
    return NativeRecommendResponse(**fit.to_dict())


@app.post(
    "/api/brain/native/generate",
    response_model=NativeGenerateResult,
    tags=["native"],
    dependencies=[Depends(require_token)],
)
async def brain_native_generate(request: NativeGenerateRequest) -> NativeGenerateResult:
    """Generate text with a trained native model (real logits, real sampling).

    Requires at least one verified (trained) model in the registry; otherwise
    returns ``ok=False`` so callers know to fall back to external providers.
    """
    from .neural.registry import DEFAULT_REGISTRY_DIR, ModelRegistry
    from .neural.training import load_model, load_tokenizer
    from .neural.inference.generator import GenerationConfig, generate
    from .neural.training.trainer import resolve_device

    try:
        registry = ModelRegistry(DEFAULT_REGISTRY_DIR)
        record = registry.resolve(request.model)
    except Exception as exc:  # noqa: BLE001
        return NativeGenerateResult(ok=False, error=f"no native model available: {exc}")

    if not record.verified:
        return NativeGenerateResult(
            ok=False, model=record.name, error=f"model '{record.name}' has no trained weights (status={record.status})"
        )

    try:
        model = load_model(record.path)
        tokenizer = load_tokenizer(record.path)
        device = resolve_device("auto")
        gen = generate(
            model,
            tokenizer,
            request.prompt,
            GenerationConfig(
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                stop_sequences=request.stop_sequences,
                seed=request.seed,
            ),
            device=device,
        )
        return NativeGenerateResult(
            ok=True,
            text=gen.text,
            prompt=gen.prompt,
            input_tokens=gen.input_tokens,
            generated_tokens=gen.generated_tokens,
            tokens_per_second=gen.tokens_per_second,
            duration_s=gen.duration_s,
            model=record.name,
            source="native",
        )
    except Exception as exc:  # noqa: BLE001
        return NativeGenerateResult(ok=False, model=record.name, error=f"native generation failed: {exc}")


@app.post(
    "/api/brain/native/route",
    response_model=NativeRouteResult,
    tags=["native"],
    dependencies=[Depends(require_token)],
)
async def brain_native_route(request: NativeRouteRequest) -> NativeRouteResult:
    """Route a task native-vs-external with an honest, auditable decision."""
    from .neural.router import NativeRouter

    decision = NativeRouter(native_registry()).route(request.task, request.model, request.text_length)
    return NativeRouteResult(
        task=decision.task,
        source=decision.source,
        model=decision.model,
        confidence=decision.confidence,
        reason=decision.reason,
    )


@app.post(
    "/api/brain/plan",
    response_model=BrainPlanResponse,
    tags=["orchestrator"],
    dependencies=[Depends(require_token)],
)
async def brain_plan(
    request: BrainPlanRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    model: OrchestratorModel = Depends(get_orchestrator_model),
) -> BrainPlanResponse:
    """Decompose a request into an ordered tool plan (no execution)."""
    return await orchestrator.plan(request, model)


@app.post(
    "/api/brain/plan/chat",
    response_model=ChatPlanResponse,
    tags=["orchestrator"],
    dependencies=[Depends(require_token)],
)
async def brain_plan_chat(
    request: ChatPlanRequest,
    settings: BrainSettings = Depends(get_settings),
) -> ChatPlanResponse:
    """Route one request deterministically — zero LLM cost, zero I/O.

    The Python port of the TS planner: intent classification, model routing,
    grounding/memory/knowledge gates and generation knobs for a message.
    """
    plan = plan_chat_request(
        request.message,
        history=request.history,
        force_grounded=request.force_grounded,
        force_model=request.force_model,
    )
    return ChatPlanResponse(
        intent=plan.intent.primary_intent,
        task_type=plan.task_type,
        priority=plan.priority,
        model=plan.model,
        grounded=plan.grounded,
        needs_memory=plan.needs_memory,
        needs_knowledge=plan.needs_knowledge,
        needs_overview=plan.needs_overview,
        temperature=plan.temperature,
        max_tokens=plan.max_tokens,
        rationale=plan.rationale,
    )


@app.post(
    "/api/brain/agent/turn",
    response_model=AgentTurnResponse,
    tags=["orchestrator"],
    dependencies=[Depends(require_token)],
)
async def brain_agent_turn(
    request: AgentTurnRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    model: OrchestratorModel = Depends(get_orchestrator_model),
) -> AgentTurnResponse:
    """Run the full agent loop: plan → execute tools → observe → finish."""
    return await orchestrator.run_turn(request, model)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {
        "service": "RelayAI Brain",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }


# Mount extensions only after the application and dependencies are defined.
from . import brain_agent  # noqa: E402,F401
from .ingest import routes as ingest_routes  # noqa: E402,F401
