# Sweep architecture and readiness

Sweep's direction is a Python personal assistant and reusable intelligence/tool
runtime: desktop operations, public-web collection, reasoning and memory.
These are separate subsystems today; there is not yet one general agent capable
of every computer or cognitive task.

## Codebase map

| Area | Role | Current use |
| --- | --- | --- |
| `sweep/` | Deterministic intent parser, skill registry, local JSON state, optional LLM intent fallback, dataset adapters | Desktop command entry point; launcher defaults to deterministic routing |
| `sweep/desktop/` | Native Qt dock, chat routing/history, task workers, permissions, provider settings, file/image inspection | Primary installed user interface; conversation and tools share one chat |
| `app/` | FastAPI, search providers, bounded HTTP fetching, extraction, browser sessions, evidence, research | Web data collection and research APIs |
| `companion/` | Provider fallback, tools, planning, agent loop, memory/RAG, ingestion, optional compute frameworks | Authenticated companion API; providers/assets depend on configuration |
| `sweep_core/` | Optional native primitives, audio/vision/scraping integration discovery | Standalone support library; imports now use its own namespace |
| `cognition/` | Claims/evidence, logic/rules, uncertainty, memory, deterministic cognitive loop | Tested reasoning building blocks; not a general language model |
| `sweep_cognitive/` | Representation, retrieval, routing, executive task state, learning and drift checks | Experimental orchestration and learned classifiers |
| `sweep_neural_mesh/` | Graph routing, specialized neurons, fusion, neural engines, plugins, training and chat | Research/optional models; readiness depends on trained assets and dependencies |
| `services/intelligence/` | Multimodal model registry, device selection, OCR/vision/intent adapters | Optional model-backed integration layer |
| `sweep_benchmark/`, `benchmarks/`, `graph_benchmark/`, `neural_eval/`, `evaluation/` | Several generations of evaluation infrastructure | Preserve datasets/results for reproducibility; scores do not establish general intelligence |
| `cpp/`, `native/` | Optional accelerators and native interface work | Not required for Python installation |
| `src/` | Remaining TypeScript web/surf implementation | Legacy source; no root package.json or complete runnable frontend |
| `companion/vendor/`, `models/` | Third-party sources/wheels and local model assets | Not bundled in the base Python wheel; do not delete as clutter |

`python scripts/audit_codebase.py` reads all first-party Python sources without
executing them, checks syntax, and emits module purposes/interfaces as JSON.
It also counts TypeScript/C++ sources. Dependency trees, virtual environments,
model data and generated output are excluded. This provides a repository-wide
structural review, not a claim that every research algorithm or third-party
dependency has received a full security audit.

## Runtime boundaries

The primary user interface is the native Qt chat dock under `sweep/desktop`.
It expands from a top-screen bar, with History and Settings in the same shell.
It directly adapts Python tools using isolated task workers and local
events/artifacts, without a browser host or localhost web server. See
[desktop integration and capability matrices](DESKTOP_PLAN.md) and
[desktop installation](DESKTOP.md). The HTTP APIs below remain optional tool APIs.

The development launch path is `setup_sweep.py` -> `.venv` -> `sweep.desktop`.
The packaged Windows application includes Python. `sweep.launcher` remains the CLI.
`sweep` remains the existing desktop controller command. No C++ compiler,
TensorFlow installation, model download or vendor framework import is required
to install the base package. Model extras are explicitly selected.

`chat.py` routes explicit statements and attachments to registered capabilities;
ordinary messages use `providers.py` and the existing provider chain. Chat context
includes recent turns and completed tool results from that thread. Model-suggested
actions pass an allowlist and require review before dispatch. Browser statements
such as `open YouTube in Brave` resolve a site URL and an installed browser
executable, then launch an argument list with shell execution disabled.

`images.py` reads only a granted image, bounds its size, extracts metadata and
uses local OCR. Optional visual reasoning sends an approved resized copy without
EXIF to the selected Ollama, OpenAI or Gemini image provider. The local setup uses
an independently installed Ollama vision model; weights are not bundled or fetched
automatically. Person identification/account matching from a photo is excluded;
visual location clues and embedded GPS are explicitly unverified. Provider keys
saved in Settings use Windows DPAPI; chat history and task artifacts remain plaintext.

The web gateway resolves and checks all returned IP addresses, connects to a
validated address, preserves TLS hostname verification, validates each redirect,
and does not forward credentials across origins. It uses a streaming byte cap,
timeouts, bounded concurrency and separate event-loop synchronization. Custom
header responses bypass the shared public cache. POST requests are not retried.
The legacy URL-prefix proxy is not used because it defeats this network policy.
Compressed responses that ignore `Accept-Encoding: identity` are rejected.

Implementation references: [HTTPX async streaming](https://www.python-httpx.org/async/)
and [OWASP SSRF prevention](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html).

Web and companion APIs require configured bearer tokens. The launcher binds to
127.0.0.1 and generates a temporary token file if none is configured. Health and
API schema pages remain public. CORS is disabled unless explicit origins are
configured. Automatic ingestion is off by default. The companion code executor
is off by default; its AST filtering and child process do **not** provide an OS
security sandbox, especially on Windows. Enable only for trusted scripts.

The model chat path defaults to shell execution disabled; enabling it still
requires approval of each generated command. The desktop controller continues
to require confirmation for shell/power commands unless its user explicitly
chooses `--yes` or changes the stored confirmation setting.

The main neural background loaders and intelligence model service coordinate
model construction through `services/model_loading.py`. This prevents overlapping
Transformer initialization contexts from interfering with process-wide Torch
hooks; inference is not serialized. Each model registry owns its own status
entries rather than mutating shared canonical entries.

The public-web gateway is stateless: it discards automatic response cookies.
This prevents IP-pinned transport URLs from sharing cookies between unrelated
hostnames on the same IP address. Explicit caller cookies follow the redirect
credential policy above.

## Remaining work and limits

- Consolidate the overlapping cognitive/agent systems behind one typed Python
  task/tool interface; avoid another parallel framework.
- Route optional ingestion, face-search and framework download HTTP clients
  through a common reviewed transport. The strengthened `app` gateway does not
  automatically cover every independent client in the repository.
- Remote browser automation is disabled by default (`ALLOW_REMOTE_BROWSER=false`).
  If enabled, its browser service needs outbound network isolation for redirects,
  scripts, subresources and DNS; top-level URL checks alone cannot secure it.
- Generated-code execution needs a real container/VM boundary and OS-enforced
  filesystem/network/resource policy before accepting untrusted code.
- Notes, histories and several memory stores are plaintext files. They are not
  password vaults. Multi-process state coordination and encryption remain work.
- Research/browse sessions are in memory; restarting the server loses them.
  Background task durability, session expiration and scheduling remain work.
- Optional checkpoints, downloaded models and vendored frameworks need separate
  provenance, dependency vulnerability and model-quality review. Old experiment
  scripts are not installed as production task entry points.
- Live search providers can change HTML, block requests or require API keys.
  Offline tests verify transport/extraction logic, not continuous provider uptime.
- Dataset adapters may require large downloads and additional libraries. Setup
  does not automatically download datasets or train models.

## Cleanup policy

Keep useful architecture/history reports, benchmark results, datasets, model
weights, vendor licenses and all existing uncommitted work. Remove only identified
scratch artifacts. This pass removes the empty root `devnull`, stale
`pytest_final.out`, `gcm-diagnose.log`, and two generated search debug HTML pages.
The generating probe script remains available. Future logs, local runtime state
and test scratch directories are ignored by Git.

## Validation of this pass (2026-10-03)

- Initial structural inventory: 661 first-party Python files, 45 TypeScript files and
  17 C++/header files; Python syntax checks and undefined-name checks passed.
- Full repository suite with cached models and Hugging Face offline mode:
  **1,768 passed, 9 skipped, 7 warnings**. Skips cover unavailable optional vendor
  assets/GPU bundles. The final cookie-isolation addition was separately checked
  with the web security suite (**26 passed**).
- A new `.venv` installed the base package without Torch, Transformers, C++ or
  model downloads. Its core suite passed **798 tests, 1 skipped**; subsequent
  targeted API, reasoning, launcher and security regressions also passed.
- A pure-Python wheel built successfully and excludes vendored frameworks,
  credentials and model weights. `pip check` found no broken requirements.
- Real Windows launcher/desktop commands, HTTPS page extraction and both local
  API services were smoke-tested. API checks covered health, rejection without
  authentication and success with the generated token.
- The native desktop and per-user Windows installer were built and smoke-tested.
  Packaged calculation and HTTPS extraction passed outside the source directory;
  the installer payload passed its CRC check and excludes credentials and models.
  Desktop tests cover real worker execution, cancellation, permission denial,
  persisted results, safe archive extraction and upgrades preserving user data.
- The exact staged first-party source was exported independently of the working
  tree and passed **820 tests, 1 skipped, 1 dependency deprecation warning** across
  the desktop, core, controller, cognition and cognitive suites. Changed Python
  files passed syntax checks; the staged credential-pattern scan found no matches.

These checks establish the tested behavior, not a guarantee of universal
automation, model accuracy or absence of all security vulnerabilities.

## Desktop task reliability milestone (2026-10-04)

- Persistent FIFO queue, explicit retry with a new task ID, cancellation before
  dispatch, and interrupted-state recovery without replaying actions on startup.
- Terminal task states reject late results; viewing history cannot redirect a
  running task's result. A parent watchdog stops blocked task workers. Failure to
  save state cannot prevent cancellation from terminating the worker.
- Bounded local IPC names and acknowledged activation report startup failures
  clearly. Background launches now show the collapsed dock without expanding its chat.
- Build metadata records source/runtime provenance; checksums are generated only
  after successful builds, with no secrets or machine-local paths included.
- Desktop regression suites: **75 passed, 1 skipped**. The skip requires Windows
  file-symlink privilege; directory-junction protections ran successfully. Tests
  include real queued workers and cross-process local IPC. Native queue/result
  rendering was also inspected on Windows.
