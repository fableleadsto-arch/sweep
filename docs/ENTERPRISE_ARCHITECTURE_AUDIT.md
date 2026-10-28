# Sweep enterprise architecture audit

Date: 2026-10-28. Baseline runtime: `3ae1258be`. Status: **Partially implemented**
(Phase 0). Governing requirements: [master directive](directives/SWEEP_ENTERPRISE_MASTER.md)
and [resource directive](directives/SWEEP_IMPLEMENTATION_RESOURCES.md).

## Scope and evidence

Ran `python scripts/audit_codebase.py` against the accessible working tree. It
parsed **729 first-party Python files**, with **zero syntax errors**, and counted
45 TypeScript, 10 C++ and seven header files. It excludes vendor trees, model
assets, environments and generated output. This is a structural inventory, not
a line-by-line security review or proof that every module works. The working
tree contains unrelated user experiments; neither those nor their results were
changed or certified by this audit. No new dependencies or datasets were installed.

Reviewed the desktop routing, task/result persistence, document parsing, OCR,
build configuration, host identity store and existing architecture/release
evidence. The earlier [architecture map](ARCHITECTURE.md),
[companion requirements](PRODUCT_REQUIREMENTS.md) and
[continuation evidence](DEVELOPMENT_STATE.md) remain relevant. Unreviewed optional
models, native implementations, vendor code and all independent HTTP clients
remain in the security/dependency audit backlog.

## Existing dependency map

```text
Windows Start menu / shortcut -> bundled Python -> native Qt dock
  chat.py + followups.py -> runtime.CAPABILITIES (18 routes)
    approval + file grants -> TaskStore -> one QProcess worker
      desktop commands -> sweep parser / skills / applications
      documents + data -> data_tools (stdlib, pypdf); images -> local OCR
      conversation + image descriptions -> installed local model service
      requested public web -> app search / HTTP / extraction / research
    result validation -> atomic saved evidence -> chat/history/export/recovery

Optional source-only text host:
  authenticated API -> SQLite identities -> FairComputePool -> local model worker

Other optional systems (not one integrated production agent):
  companion planner/tools/memory; cognition; sweep_cognitive; sweep_neural_mesh
  services/intelligence -> model assets / optional inference frameworks
```

The desktop does not require a browser UI, API server or participating peers.
Its Windows build excludes Torch, TensorFlow, Transformers, the neural mesh and
the numeric/data-science stack. Python package base dependencies still include
NumPy/Pandas/SciPy/scikit-learn: source installation and the frozen payload have
different footprints. Do not infer an installed feature from an optional extra.

## Module inventory and reuse decisions

Counts include tests and experiments beneath each component.

| Component | Python files | Existing role and enterprise decision |
| --- | ---: | --- |
| `sweep` | 39 | Native shell, command intent, task/result contracts, adapters and dataset helpers. Extend this production entry path. |
| `app` | 38 | Optional FastAPI search/research/extraction/evidence, bounded public HTTP. Reuse reviewed transport for permitted trade sources. |
| `companion` | 139 | API, agents/tools/memory, fair compute and authenticated host. Reuse narrow contracts; do not expose legacy tools through the new host. |
| `cognition` | 45 | Claim/evidence reasoning and JSONL stores. Reuse concepts after compatibility review; no automatic migration of personal memory. |
| `sweep_cognitive` | 37 | Experimental routing, executive state and learning. Separate from desktop task execution. |
| `sweep_neural_mesh` | 251 | Graph/neuron/model/training experiments. Bounded cache repairs exist; not distributed enterprise orchestration. |
| `services` | 21 | Optional model discovery/loading and intelligence adapters. Lazy, explicit adapters only. |
| `sweep_core` | 17 | Optional native/audio/vision support. Not a prerequisite for TradeOps. |
| Evaluation families | 89 | `benchmarks` 48, `sweep_benchmark` 14, `graph_benchmark` 12, `neural_eval` 12, `evaluation` 3. Preserve provenance; none establishes TradeOps accuracy. |
| Root/tests/scripts | 53 | Three root modules, 34 root test files, 16 scripts, including native packaging and structural audit. |
| Other source/assets | — | TypeScript remnants, C++/headers, models and vendored frameworks. Preserve pending provenance/dependency review. |

`pyproject.toml` declares Python >=3.12, FastAPI/Pydantic/httpx and optional
desktop/browser/model extras. Native UI uses PySide6 Essentials; PyInstaller
builds a per-user Windows installer. The existing desktop adapter reads PDF,
DOCX and text, plus bounded CSV/TSV/JSON/JSONL/SQLite. It hashes original file
bytes and returns PDF page text. Windows OCR currently returns text, not word
regions. Neither adapter provides the complete TradeOps field schema.

## Feature-status matrix

Statuses refer to the specifically described scope, not an entire product.

| Capability | Status | Evidence and remaining boundary |
| --- | --- | --- |
| Native dock, installer, chat history, recovery UI | Implemented and tested | Oct24 source regression; Oct28 packaged/installed smoke and upgrade. |
| Local conversational inference | Implemented and tested | Actual packaged greeting 24.14 s; cold latency remains unacceptable for many interactions. |
| Basic desktop launch / named browser URLs | Implemented and tested | Prior installed launch/intent tests; no general observe-act-verify agent. |
| Bounded PDF/DOCX/text/data ingestion | Implemented and tested | Real packaged workers; page text and original hashes; limited formats and extraction. |
| Image OCR | Implemented and tested | Packaged synthetic image text read locally; no layout/table accuracy benchmark. |
| Multi-document conversation comparison | Partially implemented | Up to four same-chat sources, saved excerpts; no deterministic shipment reconciliation. |
| Task approvals, cancellation, manual dependency recovery | Implemented and tested | Immutable retry requests and explicit source selection; not a general durable DAG. |
| Shared-host text processing | Partially implemented | Auth/owner queues and two local test users; no native pairing or multi-machine TLS test. |
| Neural-mesh / broad OSINT integration | Partially implemented | Source repairs and optional catalogs; 18 desktop routes are not 120 operational research tools. |
| Enterprise workspace/tenant storage | Planned | Conversation scoping and host identities are not document tenancy. |
| TradeOps structured fields, reconciliation, corrections | Planned | Needs source-linked contracts, decimal rules, matching and explicit uncertainty. |
| Side-by-side document review / CSV and Excel report | Planned | Existing source dialogs and CSV artifacts are only reusable building blocks. |
| Enterprise cloud routing, budgets, metering | Planned | Desktop is local by default; future opt-in route requires permission and cost enforcement. |
| Vision / TestLab / Evidence products | Planned | Existing image/evidence primitives and fixtures do not constitute these modules. |
| Representative customer pilot | Blocked | No authorized customer corpus, confirmed pilot participant or measured demand recorded. Independent development can continue. |
| Commercial security/reliable updates/billing | Planned | Current installer unsigned; no certified controls or verified commercial readiness. |

## Baseline performance and operating costs

Hardware queried Oct28: Intel Core i5-1335U, 10 physical cores / 12 logical
processors, 16,876,888,064 bytes reported physical memory (about 15.72 GiB).
Local inference measurements below are CPU runs; GPU suitability was not assessed.

| Observation | Actual result | Interpretation |
| --- | --- | --- |
| Current packaged greeting, one sample | 24.14 s total; five previews | No claim of a latency improvement. |
| Current packaged calculator/data/text/PDF/DOCX/OCR | 1.70 / 1.35 / 0.78 / 1.12 / 0.65 / 2.02 s | Small synthetic smoke fixtures, not a throughput benchmark. |
| Installed recovery chain | 3.14 s, two approvals, two output rows | Recovered CSV cleanup then JSON conversion; original bytes preserved. |
| Earlier Oct18 cold/warm first text | 14.14 / 2.69 s | Historical samples, not percentiles or current service guarantees. |
| Earlier model allocation | About 2.16 GB | Model allocation, not total application RAM. |
| New installer | 126,040,375 bytes | Python/runtime bundled, model weights excluded. |
| Monetary cost / idle RSS / p95 / concurrent throughput | Not measured | No unit economics, total-memory or capacity claim yet. |

Release checks and older measurements are recorded in the continuation state.
Cloud charges were not incurred for this work. Local execution still has hardware,
electricity and support costs; it must not be described as having zero cost.

## Technical debt, security gaps and missing validation

1. **Storage and authority:** personal history/tasks are local plaintext. Existing
   same-chat validation is useful but not project membership or tenant isolation.
   Host SQLite credentials authenticate text clients, not enterprise document access.
   Add workspace ownership tests before allowing shared document storage.
2. **Untrusted documents:** bounded file reads, PDF decompression limits and XML
   entity rejection exist. OCR/table extraction, MIME/signature mismatch policy,
   archive traversal coverage, malware scanning and OS-enforced resource isolation
   need explicit end-to-end tests. A worker subprocess is not an OS sandbox.
3. **External content/actions:** desktop adapters enforce approvals and grants;
   the public HTTP gateway checks targets/redirects. Other optional clients need
   coverage. Imported instructions must never gain action authority. Business
   edits/submissions/messages require a reviewed exact action, not generic trust.
4. **Durability:** completion saves result before success; retries are new tasks.
   There is no transaction spanning task/result files, project migration, general
   dependency graph, idempotency ledger or crash-safe shared-host job recovery.
5. **Extraction accuracy:** no shipment gold corpus, calibrated field confidence,
   deterministic discrepancy metrics or human correction ledger. Model comparison
   prose cannot substitute for validated source-linked fields and arithmetic.
6. **Overlapping infrastructure:** multiple agent, memory, routing and evaluation
   systems. Extend the production task boundary; do not introduce another agent
   framework to achieve a two-document deterministic comparison.
7. **Distribution/dependencies:** unsigned Windows-only release, wide optional
   dependency ranges, model/license provenance and transitive vulnerability review
   outstanding. Old capability checks failed for absent optional math/graph/vision/
   NLP libraries; this audit does not report an all-suite pass.
8. **Performance:** cold model delay, full process-tree RAM, idle footprint, failure
   recovery and realistic file-size scaling need reproducible baselines. Keep heavy
   parsers outside idle UI imports; do not load a model for arithmetic reconciliation.

## Proposed changes and migration boundaries

| Decision | Rationale | Migration / approval boundary |
| --- | --- | --- |
| Add versioned document/field/source schemas beside existing adapters | Keep page provenance and validate before saving | Additive first; retain old result readers and test old histories. |
| Add local workspace storage through the current task boundary | Explicit ownership, document versions and corrections | Design schema/versioning, backup and rollback before migrating any personal data. |
| Separate TradeOps pure rules from Qt/provider code | Decimal arithmetic and matching independently testable | New opt-in capability; preserve generic companion commands. |
| Add source navigation and review panel in native shell | Click discrepancy to actual document/page | Keep dock interaction; no browser host, automatic form submissions or decorative controls. |
| Evaluate layout parser only against existing extraction baseline | Better tables/regions may justify a heavier optional worker | Verify software/model rights and memory/latency before dependency or weight adoption. |
| Retain current orchestration for first slice | Existing approvals/retries/results work and are lightweight | LangGraph comparison below; no framework migration now. |
| Defer peer document processing | No established tenant/OS boundary | No confidential files to untrusted peers; host pairing remains optional. |

LangGraph's official documentation describes durable state and human interrupts;
these may help later branching workflows. The current small sequential workflow
already has tested approvals, isolated worker execution and explicit recovery.
**Decision:** defer adoption until a measured failure/recovery requirement cannot
be served cleanly by those contracts. Package compatibility and migration cost
have not been tested. [Official LangGraph README](https://github.com/langchain-ai/langgraph#why-use-langgraph)
supports the feature comparison, not a performance claim for Sweep.

## Next acceptance gate

Follow [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md). Complete the Phase 0
TradeOps contract, focused threat model and reproducible evaluation design; then
add a versioned source-linked document representation using existing local
extraction. Every field must retain evidence or explicitly report missing/ambiguous.
No pilot/commercial readiness claim until the remaining security, representative
data, performance and deployment gates have measured evidence.
