# Desktop integration and capability matrix

This extends the repository map in ARCHITECTURE.md. Sweep remains the same
project: the new host is `sweep/desktop`, and tool implementations are adapters
over existing modules. The legacy terminal UI is `cpp/ui/main.cpp`; TypeScript
under `src/` is not a complete desktop frontend.

## Capability matrix

| Capability | Existing | Partial | Missing | Existing module | Integration choice | Priority |
|---|---|---|---|---|---|---|
| Desktop commands | Yes | Free-text coverage | Verified general GUI agent | sweep/skills.py, parser.py | Reuse controller behind approval | P0 |
| Desktop dock/install | Added | Windows distribution | Signed releases/macOS installers | sweep/desktop/dock.py | Native Qt chat dock + bundled Python | P0 |
| Conversation | Yes | Requires provider; recent thread context | Long-context memory and autonomous workflows | sweep/desktop/chat.py, providers.py, companion/providers.py | Unified chat with reviewed tool proposals | P0 |
| Web search/scrape | Yes | Provider availability | Full browser workflows in UI | app/search, app/core/http | Direct Python adapters | P0 |
| Research/evidence | Yes | Heuristic extraction | Durable multi-step planning | app/research, app/evidence | Observable bounded task | P0 |
| Files/data | Yes | Bounded file previews in chat | Mature transformations UI | sweep/desktop/runtime.py, sweep/skills.py | Native picker/drop with exact-file grants | P1 |
| Models | Yes | Multiple loaders | Unified availability/cost router | services/intelligence/model_manager | Keep optional/lazy, adapter follow-up | P1 |
| Task state/events | Added | Persistent FIFO queue, one active worker, explicit retry | DAG scheduling/resume | sweep/desktop/tasks.py, runtime.py, window.py | Isolated worker + task JSON/events/artifacts; no automatic restart replay | P0 |
| Permission layer | Yes | Desktop task approvals added | OS-enforced isolation | Controller confirmations, desktop adapters | Explicit local/external approval | P0 |
| Memory | Yes | Several stores | Unified context/search | companion/memory.py, cognition/store.py | Preserve stores; integrate deliberately | P1 |
| Image understanding/OCR | Added | Installed OCR support; configured image model | Verified location inference; broader media workflows | sweep/desktop/images.py, ocr.py | Attachment questions in chat; local metadata/OCR and approved vision request | P0 |
| Video/audio | Yes | Individual tools | Tracking timeline/streaming voice | services/intelligence, companion/tools | Separate bounded workers | P2 |
| Browser automation | Yes | Disabled remote browser | Safe desktop-browser session workflow | app/browser, src/RelAI/browser | Playwright tool, not application host | P2 |
| Maps/satellite | Limited | Research concepts | Map workspace and data adapters | Existing tool/research primitives | Evaluate providers later | P3 |
| General autonomous agent | Several experiments | Planning loops | Verified end-to-end desktop control | companion/orchestrator, sweep_cognitive | Compose registered tools; no new agent framework | P3 |

## Architecture

Native Qt dock chat -> statement/attachment routing -> explicit permission -> task
process -> Python tool/provider -> activity events -> local artifact -> chat result.

The dock expands from a top-screen bar and contains chat, History and Settings.
It shares the established queue/worker lifecycle in `window.py`; no separate task
mode is exposed. Deterministic routes handle explicit commands, URLs and search
requests, while ordinary conversation uses the chosen provider. Models may propose
registered tasks for review; proposals are allowlisted and do not execute themselves.
Browser requests resolve website names to URLs and launch the requested installed
browser directly through an argument list, without typing a natural-language
sentence into its address bar.

Each capability has an ID, description, execution location and permission class.
Tasks carry the exact request and approval, plus file grants when applicable.
Events distinguish progress, result and error; the desktop task manager owns
queued/running/completed/failed/cancelled/interrupted state. Model output does not become a shell
command. The workspace receives artifacts, not executable UI markup. Source
content is escaped before display. Public source links require a user click.

There is no localhost HTTP bridge. QProcess workers and local task files keep
network/model work outside the UI event loop. QLocalServer uses local OS IPC only
to activate an already-running window. Idle Sweep does not load model weights.
Queued tasks receive execution approval at dispatch. Closing Sweep interrupts
waiting work; reopening only restores records and never replays actions. Retry
creates a new ID linked to the original task, with fresh permission checks. Task
storage enforces terminal states so late worker output cannot undo cancellation.
The window owns a watchdog for blocked workers and keeps active execution separate
from the history result being viewed. This is a serial queue, not an autonomous
planner or a guarantee that completed actions can be rolled back.

Provider preferences live in `providers.json`; Windows DPAPI protects keys entered
through Settings. Conversation approval can be remembered per chat. Image analysis
requires approval for each provider request, strips embedded metadata and reports
provider failures alongside the preserved local inspection result. Local OCR uses
installed Tesseract or the Windows OCR API. Ollama model discovery and starting an
installed local server do not download weights. The selected local vision setup is
`qwen3-vl:2b`, installed separately from Sweep.

## External technology matrix

Evaluated from primary project sources on 2026-10-03. “Deferred” means the project
is not bundled by this desktop work; hardware/license/model review is still
required before adopting additional capabilities.

| Project | Capability | License | Maintenance evidence | Hardware | Integration strategy | Adopt? |
|---|---|---|---|---|---|---|
| [Qt for Python](https://doc.qt.io/qtforpython-6/) | Native shell/tray/widgets | LGPL/GPL/commercial; module-specific | Current official bindings/docs; installed Essentials 6.11.2 | Desktop CPU | Dynamic PySide6 Essentials; no WebEngine | Yes |
| [PyInstaller](https://pyinstaller.org/en/stable/usage.html) | Bundle Python app | GPL with distribution exception; retain bundled notices | Current docs; installed 6.22.3 | Build on target OS | Windowed onedir app + setup executable | Yes |
| [Playwright](https://github.com/microsoft/playwright) | Browser tool | Apache-2.0 | Active official repository/docs | Browser process memory | Reuse optional browser integration; sandbox/permissions first | Existing optional, defer UI |
| [Qdrant](https://github.com/qdrant/qdrant) | Vector memory | Apache-2.0 | Active official repository | CPU/RAM varies with index | Reuse companion adapter | Existing optional |
| [DuckDB](https://github.com/duckdb/duckdb) | Local analytics | MIT | Active official repository | CPU/RAM/storage by dataset | Candidate for bounded file-data jobs | Defer until workload warrants |
| pandas/SQLite | Tables/persistence | Existing dependency; SQLite stdlib | Already used in this repository | CPU | Reuse before adding another data stack | Existing; CSV preview uses stdlib |
| Browser Use/Desktop, SearXNG | Agent/search | Individual review pending | Not reviewed in this pass | Browser/server resources | Evaluate against current controller/search | Deferred |
| SAM 2, YOLO, ByteTrack, OpenCV, PaddleOCR | Vision/OCR/tracking | Individual code/model review required | Existing adapters inspected; upstream audit pending | Model-dependent CPU/GPU | Reuse existing vision tools, add result overlays | Deferred |
| FFmpeg, yt-dlp, ExifTool | Media/metadata | Build/component-specific review pending | Upstream review pending | CPU, storage | Bounded subprocess adapters | Deferred |
| MapLibre, OSM/Nominatim, eo-learn, Earth APIs | Maps/satellite | Code and data licenses differ | Upstream/provider review pending | Dataset-dependent | Explicit provider adapters with acquisition dates | Deferred |
| Whisper/faster-whisper, VAD/TTS | Voice | Code/model review required | Existing audio modules; upstream review pending | Model-dependent | Local workers; explicit microphone activation | Deferred |

Qt's [licensing overview](https://doc.qt.io/qt-6/licensing.html) distinguishes
modules and license options. The Windows build uses the included Python setup
wizard and requires no third-party installer compiler.

## Data/source matrix

| Capability | Data needed | Existing source | Potential source/API | Local/remote | Restrictions/cost |
|---|---|---|---|---|---|
| Desktop actions | Exact user command | OS/controller | Native OS APIs | Local | Approval for actions; current user privileges |
| File inspection | Selected file | Native picker/drop | User files | Local | File grant; bounded preview; no upload |
| Search | Query | app/search providers | Configured Tavily/Exa/SearXNG, public HTML | Remote | Provider keys/quotas; results may be blocked |
| Scraping | Public URL | app/core/http | Public pages/APIs | Remote | Network validation, response limits, access controls |
| Research | Objective and sources | app/research/evidence | Search + accessible pages | Remote | Four searches/eight pages/60 seconds in desktop |
| Conversation | Prompt/recent conversation | companion/providers | Configured cloud API or local Ollama | Configurable | Explicit send approval, provider pricing/configuration |
| Image metadata/OCR | Attached image | Pillow, Windows OCR or installed Tesseract | Local file metadata/text | Local | Exact-file grant; 10 MB/25 MP limits; OCR may misread text |
| Image understanding | Attached image and question | sweep/desktop/images.py | Ollama, OpenAI or Gemini image model | Configurable | Explicit transfer approval; metadata stripped; no person identification; location unverified |
| Video | User-authorized media | Optional research modules | No integrated dock workflow | Model-dependent | Separate integration and validation needed |
| Maps/satellite | Geographic query/date | No integrated desktop source | OSM, Sentinel/Landsat/NASA candidates | Local/remote | Provider terms, rate limits and acquisition dates need review |
| Learning/datasets | Explicit dataset selection | Existing adapters in sweep | User-listed public/Hugging Face datasets | Local/remote | No automatic large downloads or training |

Future features must implement these same task/event/artifact/permission contracts
and demonstrate a real end-to-end result before they appear as ready capabilities.
