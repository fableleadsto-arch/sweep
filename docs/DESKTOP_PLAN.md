# Desktop integration and capability matrix

This extends the repository map in ARCHITECTURE.md. Sweep remains the same
project: the new host is `sweep/desktop`, and tool implementations are adapters
over existing modules. The legacy terminal UI is `cpp/ui/main.cpp`; TypeScript
under `src/` is not a complete desktop frontend.

## Capability matrix

| Capability | Existing | Partial | Missing | Existing module | Integration choice | Priority |
|---|---|---|---|---|---|---|
| Desktop commands | Yes | Free-text coverage | Verified general GUI agent | sweep/skills.py, parser.py | Reuse controller behind approval | P0 |
| Desktop window/install | Added | Windows distribution | Signed releases/macOS installers | sweep/desktop | Native Qt + bundled Python | P0 |
| Conversation | Yes | Requires provider | Unified multimodal conversation | companion/providers.py | Reuse ProviderChain | P0 |
| Web search/scrape | Yes | Provider availability | Full browser workflows in UI | app/search, app/core/http | Direct Python adapters | P0 |
| Research/evidence | Yes | Heuristic extraction | Durable multi-step planning | app/research, app/evidence | Observable bounded task | P0 |
| Files/data | Yes | Desktop preview added | Mature transformations UI | sweep/skills.py, companion/tools/data.py | Native picker + CSV table, extend existing tools | P1 |
| Models | Yes | Multiple loaders | Unified availability/cost router | services/intelligence/model_manager | Keep optional/lazy, adapter follow-up | P1 |
| Task state/events | Added | One active desktop task | DAG scheduling/resume | sweep/desktop/runtime.py, window.py | Isolated worker + task JSON/events/artifacts | P0 |
| Permission layer | Yes | Desktop task approvals added | OS-enforced isolation | Controller confirmations, desktop adapters | Explicit local/external approval | P0 |
| Memory | Yes | Several stores | Unified context/search | companion/memory.py, cognition/store.py | Preserve stores; integrate deliberately | P1 |
| Vision/OCR/media | Yes | Optional dependencies/models | Unified investigation view | services/intelligence, companion/tools | Validate adapters before UI exposure | P2 |
| Video/audio | Yes | Individual tools | Tracking timeline/streaming voice | services/intelligence, companion/tools | Separate bounded workers | P2 |
| Browser automation | Yes | Disabled remote browser | Safe desktop-browser session workflow | app/browser, src/RelAI/browser | Playwright tool, not application host | P2 |
| Maps/satellite | Limited | Research concepts | Map workspace and data adapters | Existing tool/research primitives | Evaluate providers later | P3 |
| General autonomous agent | Several experiments | Planning loops | Verified end-to-end desktop control | companion/orchestrator, sweep_cognitive | Compose registered tools; no new agent framework | P3 |

## Architecture

Native Qt window -> capability selection -> explicit permission -> task process
-> existing Python tool -> activity events -> local artifact -> native result UI.

Each capability has an ID, description, execution location and permission class.
Tasks carry the exact request and approval, plus file grants when applicable.
Events distinguish progress, result and error; the desktop task manager owns
running/completed/failed/cancelled state. Model output does not become a shell
command. The workspace receives artifacts, not executable UI markup. Source
content is escaped before display. Public source links require a user click.

There is no localhost HTTP bridge. QProcess workers and local task files keep
network/model work outside the UI event loop. QLocalServer uses local OS IPC only
to activate an already-running window. Idle Sweep does not load model weights.

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
modules and license options. An Inno Setup compiler installation was blocked by
automatic approval review; the actual build uses the included Python setup
wizard, so no Inno dependency is adopted.

## Data/source matrix

| Capability | Data needed | Existing source | Potential source/API | Local/remote | Restrictions/cost |
|---|---|---|---|---|---|
| Desktop actions | Exact user command | OS/controller | Native OS APIs | Local | Approval for actions; current user privileges |
| File inspection | Selected file | Native picker/drop | User files | Local | File grant; bounded preview; no upload |
| Search | Query | app/search providers | Configured Tavily/Exa/SearXNG, public HTML | Remote | Provider keys/quotas; results may be blocked |
| Scraping | Public URL | app/core/http | Public pages/APIs | Remote | Network validation, response limits, access controls |
| Research | Objective and sources | app/research/evidence | Search + accessible pages | Remote | Four searches/eight pages/60 seconds in desktop |
| Conversation | Prompt/recent conversation | companion/providers | Configured cloud API or local Ollama | Configurable | Explicit send approval, provider pricing/configuration |
| OCR/vision/video | User-authorized media | Optional local models | Optional configured providers | Prefer local | Asset/model readiness; explicit remote transfer if added |
| Maps/satellite | Geographic query/date | No integrated desktop source | OSM, Sentinel/Landsat/NASA candidates | Local/remote | Provider terms, rate limits and acquisition dates need review |
| Learning/datasets | Explicit dataset selection | Existing adapters in sweep | User-listed public/Hugging Face datasets | Local/remote | No automatic large downloads or training |

Future features must implement these same task/event/artifact/permission contracts
and demonstrate a real end-to-end result before they appear as ready capabilities.
