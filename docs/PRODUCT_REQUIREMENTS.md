# Sweep product requirements and implementation audit

Audit date: 2026-10-10. Baseline: `b4b338ec1`, with the current local desktop changes
identified below. This document maps the **entire original Sweep
prompt**, including its acceptance examples, to real repository code. It is a
product backlog and release checklist, not a claim that every listed feature ships.

## Current delivery slice

The current changes enforce loopback-only, installed-model inference for the
desktop, remove cloud selection from its Settings, and reject cloud-backed local
engine aliases before sending user content. Conversation, image analysis and
document questions use that boundary. Web retrieval remains an explicitly
requested external task. Model identifiers stay in collapsed advanced settings.

The dock now uses native conversation/result cards with image previews, source
cards, data profiles and task-owned file artifacts. History is searchable;
settings include reduced motion. Native welcome, Settings, working conversation,
document/data result and artifact-preview windows have been visually inspected.
Local chat streams bounded response previews into its originating task; final
answers replace previews and restore from the persisted task record. The owl
reflects thinking, searching, reading, analyzing, responding, working, success
and attention states, with accessible descriptions and no continuous idle timer.
Packaged release checks remain a separate acceptance gate.

Bounded research creates an attributed Markdown report and a JSON evidence graph
compatible with the existing cognition graph. These retain observed excerpts,
source URLs, timestamp labels, retrieval failures and scope limits. They do not
claim synthesized findings are verified or provide an interactive evidence graph.

New `documents.inspect`, `data.inspect` and `data.transform` capabilities run
through the existing permission, worker, event, task and history contracts.
Document reading supports PDF/DOCX/plain text. Local document answers are grounded
in the first 5,000 extracted characters and disclose that scope; extraction is
preserved if generation fails. Data operations support inspection, trimming and
removing blank/exact duplicate rows, selected-column deduplication, one comparison,
and CSV/TSV/JSON/JSONL conversion. SQLite inspection reads an in-memory snapshot of
ordinary tables. Transformations create a new task-owned artifact, not an edited
input. Generated SQL, joins, imputation, Excel/Parquet and arbitrary data code are
not implemented by these routes.

The data adapter's 94 passing checks cover actual document/data inputs, source
preservation, output round trips, path grants, malformed files, bounded PDF
decompression and SQLite schema hazards; one Windows symlink check is skipped
when the OS cannot create the fixture. Runtime/chat integration checks and a real
local document question were also exercised during this slice. These tests do not
substitute for the native installer smoke test or make the wider backlog complete.

## Product contract

Sweep is a Python-based, installable desktop companion. Its primary interaction
is one top-screen chat dock with Settings and History. Conversation, files,
research, analysis and actions enter through that same chat. The dock expands into
the appropriate result or investigation surface only when useful. The browser is
a tool, never the application host.

The user's later instructions refine the original specification:

- **All AI inference and processing of personal attachments must be local.**
  Cloud inference fallback is not acceptable. Existing provider keys, environment
  variables or legacy settings must not silently opt the desktop into cloud use.
- The normal interface speaks as Sweep. Runtime and model names belong in
  developer diagnostics, not routine chat, progress, onboarding or error messages.
  Local execution is still disclosed plainly when relevant.
- Public web research, downloads, scraping and user-requested browser actions
  necessarily contact external sources. That is separate from sending prompts or
  attachments to a cloud AI service. The destination and information shared must
  be reviewable. Local mode must not secretly disable requested web tools.
- Local models must be provisioned deliberately. A model download is setup, with
  size/progress/failure reporting; inference must never trigger an unannounced
  download. Missing local capability must produce an actionable local setup state.
- Polish is an acceptance requirement for every slice, not a final optional phase.
- Replace the previous installer after validating its replacement. Preserve user
  histories, grants, models, datasets, licenses, useful evaluation results and
  unrelated work; their size or age alone does not make them useless.

Private-person identification or account matching from an image or voice is not
an enabled capability. Public-profile research starts from a user-supplied name,
handle, URL or other non-biometric identifier. Media can contribute visible text,
objects, landmarks and source evidence without identifying a private person from
their face. Likewise, private-person whereabouts are not inferred or tracked.
These boundaries must remain clear in the acceptance examples rather than being
hidden behind an inaccurate “implemented” label.

## How to read the audit

- **Verified**: the bounded implementation has directly relevant regression
  coverage exercised in this audit or the recorded desktop verification. This
  does not imply all providers, models or operating systems work.
- **Partial**: reusable code exists, but the requested end-to-end desktop behavior
  is incomplete, unverified, disconnected, or narrower than the requirement.
- **Missing**: no complete implementation was found in the inspected first-party
  product paths. A model name, dependency, vendored framework or roadmap entry
  alone is not an implementation.
- **Boundary**: the original example includes biometric identification or private
  location tracking; the supported alternative is recorded explicitly.

Priorities describe delivery order: **P0** fixes the current local dock contract;
**P1** completes useful file/data/research workflows; **P2** composes multimodal
investigations and voice; **P3** adds verified general automation and geographic
workflows. They are not a decision to abandon later work.

## Architecture, modules and dependencies

| Layer | Real implementation | Reuse decision and gap |
|---|---|---|
| Desktop host | `sweep/desktop/__main__.py`, `dock.py`, `platform.py`, `owl.py` | Keep native Qt, OS IPC, tray and top-screen dock. No replacement web app. |
| Chat and presentation | `sweep/desktop/chat.py`, `presentation.py` | Keep one intent/attachment entry point and thread store. Extend result cards and follow-up context. |
| Worker/task lifecycle | `sweep/desktop/window.py`, `tasks.py`, `runtime.py` | Keep isolated workers, persisted requests/events/results and terminal-state rules. Add task dependencies to this system rather than another queue. |
| Intent and desktop skills | `sweep/parser.py`, `controller.py`, `skills.py` | Reuse deterministic commands and browser resolution. Add observations and verification around future GUI actions. |
| Web collection | `app/search/`, `app/core/http.py`, `app/extraction/`, `app/scraping/`, `app/browser/sessions.py` | Reuse bounded public HTTP/extraction. Browser sessions are not yet a verified native-browser agent. |
| Research and source evidence | `app/research/engine.py`, `app/evidence/`, `app/core/types.py` | Reuse bounded searches, source records and excerpts. Persist the research plan/results under a desktop task and map to the existing evidence graph. |
| Tool and capability systems | `companion/orchestrator.py`, `companion/contracts.py`, `companion/capabilities.py`, `sweep_neural_mesh/registry/capability_registry.py` | Several registries exist. Adapt their useful schemas to the desktop registry; do not expose every optional tool automatically. |
| Reasoning and evidence | `cognition/evidence_graph.py`, `schema.py`, `store.py`, `uncertainty.py`, `verification.py` | Reuse typed entities, relations, claims, supporting/refuting evidence and source provenance. They do not by themselves implement web or media investigation. |
| Memory/context | `companion/memory.py`, `cognition/memory.py`, `sweep/desktop/chat.py` | Preserve stores. Start with explicit thread/task context; integrate authorized indexed memory with deletion and provenance. |
| Local AI/model management | `sweep/desktop/providers.py`, `companion/providers.py`, `services/intelligence/model_manager/` | Reuse inference and lazy-loading adapters, enforce local-only desktop selection, and keep model availability distinct from capability readiness. |
| Images/OCR | `sweep/desktop/images.py`, `ocr.py`, `companion/tools/vision.py`, `services/intelligence/document/ocr_engine.py` | Keep permission-bound image inspection and native OCR. Normalize detection/OCR/embedding outputs before exposing advanced adapters. |
| Documents/data | `sweep/desktop/data_tools.py`, `companion/tools/data.py`, `sweep_cognitive/perception/document.py` | Bounded local adapter now connects to chat and task artifacts. Existing companion tool profiles/aggregates/plots; its “clean” docstring does not implement cleaning. Existing perception code parses text/structure, not a complete PDF reader. |
| Video/audio | `services/intelligence/vision/opencv_engine.py`, `sweep_core/integrations/audio.py` | Real frame extraction/metadata and transcription building blocks exist. Add grants, installed-only assets, time budgets, timestamps and task artifacts before exposing them. |
| Optional scientific/data tools | `companion/tools/numeric.py`, `symbolic.py`, `ml.py`, `vectors.py`, `sweep/datasets/` | Reuse behind typed requests and availability checks. Dataset downloads/training are explicit tasks, not desktop startup work. |
| Legacy/experimental systems | `src/`, `cpp/`, `sweep_cognitive/`, `sweep_neural_mesh/`, `companion/vendor/` | Preserve useful algorithms/assets/licenses; no blanket claim that research modules are production features. |
| Distribution | `scripts/build_desktop.py`, `scripts/desktop_installer.py`, `setup_sweep.py`, `pyproject.toml` | Keep Windows per-user setup and bundled Python; validate the current dock from the built payload outside the repository. |

The core dependency set contains the Python web/extraction and scientific stack;
the desktop extra adds Qt, Pillow and bounded PDF reading through pypdf. Advanced
browser, vision, speech and model frameworks are optional extras. A manifest entry does not establish that an asset
exists on the user's machine, has the correct license, runs without downloading,
or fits memory. Optional imports must stay out of idle desktop startup.

The intended shared contract remains:

`chat objective -> typed capability -> permission/grants -> task -> worker -> events -> evidence/artifacts -> native result component`

## Complete capability matrix

### Companion, desktop presence and visual system

| Requirement family | Status | Existing module/evidence | Concrete completion work | Priority |
|---|---|---|---|---|
| Natural conversation plus actions in one chat | Partial | `chat.py`, `providers.py`, `dock.py`; local inference, thread context and reviewed task proposals | Retain actual tool observations in context; handle compound objectives without inventing completion. | P0 |
| Statement understanding, including “open YouTube in Brave” | Verified, bounded | `parser.py`, `skills.py`, `tests/test_browser_commands.py` | Preserve requested browser and actual URL. Expand intent tests using real user phrasing; add action verification for complex workflows. | P0 |
| Native installation, launch, tray, shortcut, notifications, optional startup | Partial | `__main__.py`, `platform.py`, `window.py`; desktop/instance/build tests | Rebuild and smoke-test the current dock installer; verify upgrade, startup toggle and second-instance activation. Windows is the tested platform. | P0 |
| Lightweight top dock, command bar, chat, history/settings, contextual expansion | Partial | `dock.py` implements collapse/expand, hover, pin, drag, resize and dialogs | Finish focus/keyboard/multi-monitor/DPI behavior; keep task result expansion within the same conversation. | P0 |
| Calm materials, typography, spacing, radii, shadows, color, density | Partial | `DOCK_STYLE`, native presentation cards | Establish consistent tokens; inspect real native windows at multiple scale factors; eliminate generic technical dumps and clipped controls. | P0 |
| Spring movement, smooth expansion, continuity, contextual morphing and hover | Partial | Dock opacity animation and hover expansion | Add short meaningful transitions with reduced-motion setting; animate state changes only when they communicate progress. Map/timeline transitions depend on those views. | P0/P2 |
| Original expressive owl: idle/listening/thinking/searching/analyzing/working/reading/observing/success/uncertainty/error/sleep | Partial | `owl.py` draws task-driven thinking/searching/reading/analyzing/responding/working/success/attention states with accessible descriptions | Add listening/observing/sleep only when those real workflows exist; fuller expression and restrained motion remain unfinished. | P0 |
| Never a blank/generic working state | Partial | Worker stages, task titles, counts and bounded local response streaming; per-task preview isolation and late-event tests | Extend detailed stages to future capabilities; a stalled task exposes cancel/retry and its last completed stage. | P0 |
| Accessible keyboard, focus, contrast, screen-reader names, reduced motion | Partial | Some accessible names and Enter/Shift+Enter handling | Audit every interactive control, tab order, focus return, shortcuts, contrast and screen-reader output; add explicit reduced-motion support. | P0 |
| Developer diagnostics without private chain-of-thought | Partial | Task requests/events/results and stored errors | Put tool IDs, timing, models, cache and raw data behind a diagnostics view. Normal chat remains product language; never render hidden reasoning. | P1 |

### Local intelligence, architecture and task execution

| Requirement family | Status | Existing module/evidence | Concrete completion work | Priority |
|---|---|---|---|---|
| Entirely local AI; no cloud fallback or vendor branding | Implemented in current desktop slice; release validation pending | `providers.py` rejects remote addresses/aliases, strips cloud credentials, checks model metadata; `images.py` shares the gate | Exercise packaged chat/image/document tasks with poisoned environment/legacy config and verify no remote AI calls. Reusable developer APIs retain their separate configuration. | P0 |
| Tool metadata: ID/description/inputs/output/permission/provider/location/latency/cost | Partial | Desktop `Capability` plus companion `ToolSpec` and registries | Consolidate schemas through adapters; add validated input/output contracts, availability and estimates. Internal provider fields remain diagnostics. | P1 |
| Capability registry across vision/audio/data/web/maps/files/computer | Partial | Small desktop registry; larger disconnected companion/mesh registries | Register only working adapters under stable IDs. A capability becomes available only after dependencies, local assets and required permissions are checked. | P1 |
| Shared task state, cancellation, retry, restart recovery | Verified, bounded | `tasks.py`, worker lifecycle; task/queue regression suites | Preserve no-replay restart recovery. Extend to child tasks while keeping late results from resurrecting cancelled work. | P0/P1 |
| DAG/task graph with independent work in parallel | Missing from desktop | Companion bounded agent loop and experimental graph primitives; desktop is FIFO | Add parent/dependency IDs and bounded child scheduling in existing task store; independent metadata/OCR/source tasks can run concurrently; failure preserves siblings. | P1 |
| Observe/plan/select/execute/observe/correlate/verify/present loop | Partial | `companion/orchestrator.py`, research loop, cognition verification | Adapt a bounded local planner to registered desktop tools, ground each next step in an observation, attach verification and stop conditions, and expose the plan as activity. | P1/P3 |
| Artifact system, first-class outputs and context | Partial | Task `result.json`, named data/report artifacts with hashes, native cards, inert text/image preview and Save a copy | Add typed MIME/source manifest and reuse in follow-ups; audio/video preview remains missing. Task-owned path checks prevent arbitrary file reads. | P1 |
| Context, long-term memory and unified retrieval | Partial | Thread store and `companion/memory.py` file/vector stores | Separate chat context from authorized indexed files; expose scope/deletion and citations; avoid leaking one thread's attachments to another. | P1 |
| Local model routing by task and hardware | Partial | Device/cache/loader registry; separate chat and image settings | Select available local chat/OCR/detection/embedding/speech models by capability, memory and latency; unload idle models; no fallback to remote inference. | P0/P2 |
| Low idle cost, lazy load, CPU/GPU, workers, parallelism, streaming/cache | Partial | Workers, timeouts, lazy imports and model cache infrastructure | Measure idle CPU/RAM; stream meaningful events; enforce model memory budgets; cache by input hash/version; bound parallelism and cancellation of child work. | P0/P1 |
| Permission classes READ/WRITE/EXTERNAL/DESTRUCTIVE/ACCOUNT/LOCATION/IDENTITY_RESEARCH/COMPUTER_CONTROL | Partial | Worker approval and exact-file grants; some permission classes | Add scoped typed grants per resource/action/task; independent permission checks at dispatch; distinguish local inference from external research and file writes. | P0/P1 |
| User controls for web/profile/media/location/maps/voice/computer/connectors/automatic search | Partial | Existing desktop preferences and approval dialogs | Persist capability-specific policy with allow once/task/explicit remembered scopes; show what input leaves the device; keep microphone activation explicit. | P1/P2 |
| Access-control, authentication and credential boundary | Partial | API bearer authentication, hardened public HTTP; desktop ignores preserved legacy encrypted credentials | Review each adopted legacy adapter independently; do not inherit bypasses, raw code execution, unverified TLS or unrestricted network clients. | P0 |
| Failure UX with partial work, unavailable/contradictory counts and retry/continue | Partial | Task errors/retry, research partial results | Keep completed evidence/artifacts visible when later steps fail; retry only requested failed children; expose why a source was unavailable or contradictory. | P1 |

### Files, documents, structured data and creation

| Requirement family | Status | Existing module/evidence | Concrete completion work | Priority |
|---|---|---|---|---|
| Files/folders locate, inspect, search, open, create, rename/copy/move/index/organize | Partial | `skills.py` open/list/create and desktop file grants | Add bounded folder grants and indexed search; typed reviewed file plans for writes/moves with collision handling and artifact receipts. No unrestricted generated shell commands. | P1 |
| PDF, DOCX, plain text and logs | Verified, bounded extraction; partial document understanding | `data_tools.inspect_document` and `documents.inspect` route; PDF/DOCX fixtures and local question smoke test | Extend beyond first-5,000-character questions to cited full-document retrieval; add scanned-page OCR. PDF previews are limited to 100 pages/80,000 characters. | P1 |
| CSV, JSON, JSONL, structured text | Verified, bounded | `data_tools.py` loaders/profiles, chat routes and artifact cards; preservation/round-trip tests | Extend beyond 20 MB/20,000 rows/128 columns only with streaming and resource tests. Support richer nested-data exploration without flattening values silently. | P1 |
| Excel, Parquet, XML, SQL and SQLite | Partial | New read-only SQLite adapter; XML text/doc parsing; pandas elsewhere | Add bounded Excel/Parquet readers/writers and typed XML table extraction. SQL uses read-only inspection/allowlisted operations rather than generated arbitrary queries. | P1 |
| Cleaning, normalization, deduplication, filtering, conversion | Partial; bounded clean/deduplicate/filter/convert verified | Finite-operation data adapter and `data.transform`; new artifact, source hash and row-count receipts | Normalization/imputation and compound filters remain missing; add typed operation contracts and previews rather than executing generated code. | P1 |
| Joining/merging, aggregation/statistics, schema inference, anomaly detection | Partial | Companion profiles, optional grouping, numeric/ML tools | Add multi-file grants, explicit join keys/type, row estimates and previews; detect anomalies with documented method; integrate local results into dataset cards. | P1 |
| Reports/tables/spreadsheets/databases/datasets/documents/charts/JSON outputs | Partial | JSON task export, companion plot, data artifacts and attributed Markdown research report plus JSON evidence graph | Add remaining format adapters; synthesized findings and charts must retain source/transform evidence. Office document generation remains missing. | P1 |
| Resource types: database/API/media/model/website/cloud resources | Partial | Several independent integration adapters | Represent resource IDs and grants uniformly; connect authenticated resources only when configured/authorized; avoid advertising unreachable connectors. | P2 |

### Web, public information and evidence

| Requirement family | Status | Existing module/evidence | Concrete completion work | Priority |
|---|---|---|---|---|
| Public web search | Verified, bounded | `app/search/engine.py`, search-readiness tests | Preserve time budget/fallback/source URLs; render candidate counts/source cards and actionable unavailable states; live provider availability remains variable. | P0/P1 |
| HTML/API scraping, text/metadata/links/tables | Partial | `app/core/http.py`, `extraction/page_data.py`, `scraping/` | Show extracted structure and downloadable records; add explicit pagination budget, linked-resource policy and API preference. | P1 |
| Web navigation, click/type/forms/downloads/repetitive workflows | Partial | `app/browser/sessions.py`, legacy browser modules | Implement an authorized local browser session adapter with observations before/after actions, page-state checks, download grants and confirmation at consequential submission. Current remote-browser helper is not sufficient. | P3 |
| Authorized authenticated web sessions/accounts/devices | Missing as dock workflow | Optional API/auth primitives; no integrated session handoff | User selects session/resource; isolate credentials, scope access and show action targets. Never bypass a login, private account or access control. | P3 |
| User-directed research from files/media/web/public databases/datasets | Partial | Research engine currently starts from text and public search | Compose attachment-derived non-biometric clues into explicit external queries, then correlate retrieved sources in the same task; record the exact information shared. | P1/P2 |
| Search progress: queries, candidates, relevant/analyzed counts, extraction and relationships | Partial | Research action events and source/evidence collections | Emit stable stage/count/source events and update source cards incrementally; counts come from actual work, never invented animation. | P1 |
| Public social presence by supplied name/handle/URL | Partial | General search; legacy social parsing under `face_search/` | Dedicated public-profile adapter with platform/source URL, retrieval time, exact handle and explicit supporting/contradicting links. Do not infer accounts from appearance. | P1/P2 |
| Profiles across Instagram/X/LinkedIn/GitHub/Facebook/Reddit/YouTube/TikTok/blogs/forums/portfolios/directories | Partial | Search can return indexed public pages | Treat platform availability independently; only return accessible observed accounts; authenticated/private/unavailable pages remain explicit gaps. | P2 |
| Unified web/file/vector/OCR/image/video/metadata/map/project/connected search | Missing as a unified desktop capability | Separate web search, memory/vector tools and local file tools | Implement a scoped query planner and normalized hits carrying source kind, content hash, excerpt and timestamp; rank without hiding each hit's origin. | P2 |
| Source/timestamp/entity/relation/confidence/evidence provenance | Partial | `app/core/types.py` source/excerpt schemas; cognition evidence model | Map every desktop artifact and research claim into stable evidence records, keep capture/retrieval/publication times distinct, and never invent source dates. | P1 |
| Interactive evidence graph with confirmed/strong/likely/possible/weak/contradictory states | Partial | `cognition/evidence_graph.py` persists typed relations and support/refutation | Adapt research/media artifacts to graph; show source-backed edges with confidence explanation, contradictions and human confirmation, rather than treating a score as proof. | P1/P2 |
| Media origin, uploader/creator/owner/account/person distinctions | Missing from desktop | Search/extraction/evidence primitives only | Define separate claim types; compare source timestamps and attribution statements; report earliest accessible evidence, not an unprovable global origin. | P2 |

### Images and cross-media research

| Requirement family | Status | Existing module/evidence | Concrete completion work | Priority |
|---|---|---|---|---|
| Image metadata, OCR and bounded local inspection | Verified, bounded | `images.py`, `ocr.py`, image regression tests | Add OCR region coordinates and improved nested metadata display; retain malformed-file, grant and image-size checks. | P0/P1 |
| Camera/device: make/model/lens/software/encoder/time/GPS/resolution/EXIF/XMP/container | Partial | Curated EXIF, dimensions, hashes and GPS in `images.py` | Add bounded XMP/container adapter; show original field, provenance and editable-metadata caveat; absent fields remain unknown. | P1 |
| Local description, classification, scene understanding, landmark/logo clues and object relationships | Partial | Local multimodal adapter; optional model registry | Validate on real images and explicit uncertainty cases; separate visible observation from inferred label and verified source evidence. | P0/P1 |
| Detection and segmentation | Partial | Optional detection model registry/loader; OpenCV/vision helpers | Add installed-only detection/segmentation adapter with normalized boxes/masks, confidence and model provenance; no download on inference. | P2 |
| Prominent image, clickable overlays, OCR/sign/logo/vehicle/building/object inspector | Partial | Image thumbnail/card work; basic OCR lacks regions | Add shared coordinate system, detection/OCR overlays, selected-object crop grants and contextual local/search actions. | P1/P2 |
| Reverse search: hashes/embeddings/local index/web/candidates/rerank/source analysis | Partial, legacy code not integrated | CLIP model loader/vector tools and legacy reverse-search providers | Build non-person media similarity adapter with exact/perceptual hashes, local index and explicit external-query/upload consent; review existing providers before reuse. | P2 |
| Exact/cropped/resized/screenshot/edited/meme/visually similar image matches | Missing as tested workflow | Embedding/model assets are building blocks only | Create transformed-image fixtures and measure retrieval/reranking; show side-by-side candidate evidence and known false positives. | P2 |
| Earliest image occurrence/source/repost chain/creator or owner claims | Missing as tested workflow | Search, reverse-search candidates and evidence primitives | Extract publication evidence from actual pages, distinguish retrieved vs published dates, deduplicate reposts, and show timeline/claim confidence. | P2 |
| Image place finding: GPS/OCR/signs/roads/buildings/terrain/language/shadows/maps/satellite | Partial | Embedded GPS and unverified local visual clues | Compose OCR/local analysis/public-source search/geocoding; rank candidates with supporting and contradicting clues; display hypotheses on a map. | P2/P3 |
| Research this building/object; search public sources from visible media clues | Partial | Image analysis and web research separate today | Generate reviewable object/text/landmark queries from attachment observations and link every source back to the originating crop/clue. | P1/P2 |
| Identify a person or discover their accounts from an image | Boundary | Desktop intentionally blocks image-to-person/account identification | Support supplied-name/handle research and non-sensitive visible scene analysis; do not activate legacy face-identification providers. | — |
| Compare the appearance of people in supplied files | Partial, bounded alternative | Local vision can describe visible attributes | Offer side-by-side non-sensitive visible comparisons without asserting identity, inferring sensitive traits or matching the person to public accounts. | P2 |

### Video, tracking and object location

| Requirement family | Status | Existing module/evidence | Concrete completion work | Priority |
|---|---|---|---|---|
| Video metadata and keyframes | Partial | `OpenCVEngine.get_video_info` and `extract_frames` | Add exact-file grants, timestamped frame artifacts, seek/time/frame limits and missing-codec errors; inspect inside the dock. | P1 |
| Scene/action/motion/object/OCR/entity/transcript analysis | Partial building blocks | Video frames plus image/OCR and optional speech models | Start with timestamped scene/keyframe observations; run installed local adapters; show limitations instead of claiming comprehensive action recognition. | P2 |
| Interactive video timeline with clickable occurrences | Missing | No native timeline component | Build timestamped keyframe/track/transcript rows and local playback seek; each occurrence opens source frame and evidence. | P2 |
| Tracking within frames/clips and across many videos/images | Missing as product workflow | Detection/embedding primitives, no integrated tracker | Adapt a tracker, then local embedding candidate retrieval and reranking with human confirmation; distinguish a track from proven object identity. | P2 |
| “Find this motorcycle in these 50 files” / object locator | Missing | File and model primitives are disconnected | Reference-object selection + folder grant + resumable bounded indexing + occurrence timeline; evaluate misses and lookalike false positives. | P2 |
| Video origin through frames/hashes/OCR/audio/subtitles/logos/public sources | Missing | Frame extraction and web research primitives | Link frame/audio clues to public results, compare uploads and attribution claims, and build an evidence timeline. | P2 |
| Video uploader/creator/copyright owner/account owner/person distinctions | Missing | Generic evidence relationships can represent these | Preserve each as a separate attributed claim with source/date/confidence; never equate uploader with copyright owner. | P2 |
| Video place finding from landmarks/text/roads/architecture/terrain/maps | Missing as complete workflow | Frames + local image clues are reusable | Sample distinct scenes, aggregate consistent/conflicting clues, geocode candidate places and show temporal scene-to-map links. | P2/P3 |
| Person tracks and public identification from video | Boundary/Partial | Optional detection can locate visible people; no dock tracks | Non-identifying person/object tracks may support scene analysis. Do not expose biometric identity/account lookup from those tracks. | P2 |
| Object occurrences across documents/projects/connected/public sources | Missing unified workflow | Separate resource and search adapters | Normalize occurrence source, page/frame/time/crop and confidence; share the object representation and occurrence UI across media. | P2 |

### Maps, Earth observation and location evidence

| Requirement family | Status | Existing module/evidence | Concrete completion work | Priority |
|---|---|---|---|---|
| Geocoding/reverse geocoding, pins/routes/polygons/layers/distance/elevation | Missing integrated implementation | Coordinates from image metadata only | Add licensed source adapters, local distance calculations and native-hosted map view; cache permitted data; show source and precision. | P3 |
| Public geographic datasets and satellite context | Missing | Dataset/research abstractions only | Add explicit region/date/source requests and bounded downloads, distinguish online tile/data access from local inference, and store acquisition dates. | P3 |
| Satellite before/after: construction/terrain/flooding/vegetation/water/fire/urban/environmental change | Missing | Image/data primitives are not geospatial analysis | Read georeferenced rasters, align bands/resolution/cloud masks, compute validated change measures and render synchronized before/after views with uncertainty. | P3 |
| Sentinel/Landsat/NASA and configured licensed providers | Missing adapter suite | No desktop Earth-observation provider | Review data license, credentials, quota, coverage and acquisition timestamps; add one complete provider workflow before broadening. | P3 |
| Person location research from historical/current public or authorized information | Boundary/Partial alternative | Generic source timestamps and map ideas only | Support user-authorized self-location and public event/place research with dated evidence; do not infer or track private-person whereabouts. | — |

### Speech, recordings and spoken interaction

| Requirement family | Status | Existing module/evidence | Concrete completion work | Priority |
|---|---|---|---|---|
| Local speech-to-text from attached audio | Partial | Vosk/Whisper/faster-whisper adapters | Accept installed local assets only; add file grants, supported-format conversion, budgets and timestamped transcript artifacts. | P1/P2 |
| Timestamps/language/speaker segments/searchable transcript/keywords | Partial building blocks | Whisper/Vosk optional models; diarization model registry | Preserve segment/word timestamps instead of dropping them; label speakers anonymously; local transcript search and waveform-linked excerpts. | P2 |
| Compare authorized recordings for likely same speaker | Missing as desktop workflow | Optional audio representations/diarization loader | Treat comparison as consent-based probabilistic analysis, report sections and uncertainty; no identity inference or lookup from voice. | P2 |
| Shortcut -> microphone -> streaming transcription -> intent/action | Missing | Desktop shortcut exists; no streaming microphone workflow | Explicit recording state, local VAD/STT, partial/final transcript, correction and cancellation; return to the same chat. | P2 |
| Low latency, punctuation, corrections, context and interruption | Missing | No complete voice session controller | Add session state machine, measure time-to-first-text, retain user edits, and cancel current generation/action proposal when interrupted. | P2 |
| Natural local TTS, sentence streaming and stop when user speaks | Missing | No integrated native playback/TTS route | Select licensed installed local voice, stream sentence audio, expose mute/stop, and pause immediately on microphone speech or user cancellation. | P2 |

## Required interactive components

Components are reusable views over the same task/evidence/artifact contracts.
Adding a new page or demo for each capability would violate the product direction.

| Requested component | Present state | Required integration |
|---|---|---|
| `SearchProgress`, `SourceCard` | Activity rows and source-card work | Actual query/source lifecycle, counts, source status and evidence excerpts. |
| `MediaViewer` | Granted image thumbnail | Zoom, orientation, local playback and source-artifact links. |
| `DetectionOverlay`, `ObjectInspector` | Missing | Boxes/masks/OCR coordinates, selected crop and contextual actions. |
| `PersonInspector` | Missing | Non-identifying visible attributes/source context; supplied-name research separate from face data. |
| `TrackingTimeline` | Missing | Video frames, object tracks and transcript segments sharing timestamps. |
| `EvidenceGraph` | Graph backend exists | Interactive typed edges, source detail, contradictions and confidence legend. |
| `MapExplorer`, `SatelliteViewer` | Missing | Source/date/precision-aware map and georeferenced comparison. |
| `DatasetViewer` | Preview/profile card work | Scrollable schema/table/profile plus explicit transform preview and output. |
| `TaskGraph` | FIFO task activity only | Parent/child dependency status, cancel/retry and preserved partial results. |
| `FileArtifact` | Named data/report artifact cards, task-owned paths, inert preview, Save a copy | Typed MIME/source manifest and follow-up attachment; audio/video preview remains missing. |
| `AudioWaveform`, `TranscriptViewer` | Missing | Local playback, timestamped transcript and anonymous speaker sections. |
| `ComparisonViewer`, `ImageMatchGrid` | Missing | Side-by-side evidence, match region, retrieval method and uncertainty. |
| `ProfileMatchGrid` | Missing | Public URL/name/handle evidence and contradictions, with no fabricated accounts. |
| `OwlState` | Original vector owl, task-driven semantic states and accessible descriptions | Full requested expression set depends on connected voice/observation workflows; richer restrained animation remains unfinished. |

## Technology decisions and adoption gates

This is the complete candidate inventory from the original prompt. Existing
license labels below refer to repository manifests or the earlier
[desktop technology audit](DESKTOP_PLAN.md), not a fresh upstream legal or
maintenance review. A candidate with an unverified current license, model license
or maintenance state is **not approved for bundling** merely because it is named.
Keep exact versions, notices, asset checksums and setup sizes with each adopted
adapter. Performance must be measured on the target machine before calling it ready.

| Project/family | Capability | License/maintenance evidence in this repository | Hardware | Integration decision |
|---|---|---|---|---|
| Qt/PySide6, PyInstaller | Native shell/distribution | Existing desktop audit and built versions; retain applicable notices | Desktop CPU | Continue existing host and installer. |
| Browser Use, Browser Use Desktop | Browser/computer agents | No completed adoption review | Browser and local model dependent | Compare with existing orchestrator; adopt only an adapter that satisfies local inference, observation and permission contracts. |
| Playwright | Browser navigation/actions | Existing optional dependency; earlier audit records Apache-2.0 | Browser process | Reuse Python integration for authorized local sessions; current remote helper is insufficient. |
| SearXNG | Search aggregation | Provider concept, no bundled server review | Additional local/server process | Evaluate against working search adapters; do not add a required service just to rename search. |
| SAM/SAM 2 | Segmentation | Code/model license and current maintenance review missing | Model dependent CPU/GPU/RAM | Add only after detection result contracts and overlays exist. |
| Ultralytics YOLO | Object detection | Registry declares AGPL-3.0 for selected model integration; packaging review needed | Selected checkpoint dependent | Existing loader is reusable after installed-only asset checks and distribution decision. |
| ByteTrack/other trackers and re-identification | Tracking/non-person object matching | No integrated implementation or current review | Detector/video throughput dependent | Compare on known object clips and false positives before adoption. |
| OpenCV | Image processing/video frames | Existing adapter; registry declares Apache-2.0 | CPU; optional acceleration | Immediate reuse for bounded video metadata/keyframes. |
| PaddleOCR | OCR regions/languages | Candidate, no completed adoption review | Model/language dependent | Compare with working native OCR; add when it improves region/language coverage. |
| FAISS, Qdrant | Vector/similarity search | FAISS optional dependency; Qdrant tool/memory adapters and earlier audit | Index size dependent CPU/RAM/storage | Start with one local index behind common search contract; avoid two mandatory services. |
| FFmpeg | Audio/video normalization and metadata | No bundled-build license review; codecs affect distribution | CPU/storage | Bounded argument-list subprocess adapter; validate exact binary and codec build. |
| yt-dlp | Authorized media acquisition | No adopted desktop adapter/current review | Network/storage | Optional explicit download tasks; no private-content/access-control bypass. |
| ExifTool | Broader media metadata | No adopted desktop adapter/current review | CPU | Compare to current curated Pillow metadata; use bounded output/process and trusted binary. |
| MapLibre, OpenStreetMap, Nominatim | Maps/geocoding | Code/data/service terms differ; no complete adoption review | Tile/data dependent | Native-hosted map only; select permitted tile/geocoder source and rate policy. |
| eo-learn, Sentinel/Landsat/NASA providers | Earth observation | No complete adapter/license/quota review | Raster volume dependent | One dated, georeferenced before/after workflow before adding provider breadth. |
| Whisper, faster-whisper, VAD | Local transcription/voice | Existing audio adapters; local-only loading not yet guaranteed | Model dependent CPU/GPU/RAM | Adapt explicit installed paths and timestamped output; no first-use network downloads. |
| Local TTS | Spoken responses | No selected adapter/voice license review | Voice dependent | Choose by measured latency and voice license; cloud TTS does not meet current requirement. |
| pandas, SQLite | Data profiles/transforms/persistence | Existing dependencies/stdlib and working modules | Dataset dependent CPU/RAM | Reuse first; bounded local data adapter avoids importing heavy stacks for simple operations. |
| Polars, DuckDB, Arrow | Larger data/Parquet/SQL workloads | Candidates; DuckDB included in earlier audit, not a desktop integration | Dataset dependent CPU/RAM/disk | Introduce when a measured workload needs them, under the same safe data/artifact contracts. |

## Data and source matrix

| Capability | Data needed | Existing source | Next source/adapter | Local/remote | Restrictions/cost and provenance |
|---|---|---|---|---|---|
| Conversation/planning | Current thread and observed tool results | Local chat/task store | Installed local text model | Local | No cloud inference; cache/context deletion and model availability are explicit. |
| Document/data work | Selected files and explicit operation | Native file picker/drop, bounded loaders | Excel/Parquet/PDF extensions | Local | Exact grants, size/time limits, source hash, new output file. |
| File/project search | Authorized folder contents/index | Existing file and memory modules | Bounded local index | Local | Scope excludes unrelated/private resources without grants; deletion removes indexed content. |
| Image analysis | Attached pixels/metadata | Pillow/native OCR/local vision | Installed detection/segmentation/embedding assets | Local | Model license, memory and pixel limits; observations are not verified identities/locations. |
| Image similarity/origin | Reference image, hashes/candidates/page dates | Local model/vector and legacy search primitives | Local index; reviewed public reverse-search providers | Local plus explicit remote research | External image transfer must be separately disclosed; dates/creator claims need sources. |
| Public web/profile research | User query/name/handle/URL or approved visual clue | Existing search and safe HTTP | Public indexed pages/APIs and authorized sessions | Remote retrieval, local reasoning | Actual accessible pages only; rate limits/keys/blocked sources; no fabricated profiles. |
| Video understanding | Authorized video files | OpenCV frame/metadata adapter | Local detector/OCR/STT/tracker | Local | Frame/time/decode budgets, timestamps, anonymous tracks and candidate confidence. |
| Audio/voice | Authorized file or explicitly activated microphone | Existing offline STT adapters | Installed-only STT/VAD/TTS/segmentation | Local | Recording state, consent, timestamps and no biometric identity claims. |
| Maps/geolocation | Coordinates/clues and map data | Embedded GPS only | Licensed tiles/geocoder/geographic files | Local computation plus requested remote data | Precision, source, query disclosure, cache terms and historical/current distinction. |
| Earth observation | Region, dates, georeferenced rasters | No desktop source | Reviewed Sentinel/Landsat/NASA/licensed source | Download plus local analysis | Acquisition date, band/resolution/cloud metadata, license and quota. |
| Computer/browser actions | Authorized windows/session and observations | Deterministic launcher, optional browser module | Local OS/browser observation/action adapter | Local control, websites as requested | No assumed click success; exact target/action/verification; consequential submission review. |
| Research evidence | Excerpts/artifacts/entity claims/source relationships | `app` evidence and `cognition` graph | Shared adapter | Local storage | Retrieval/publication/capture dates distinct; support/refute/unknown confidence explicit. |
| External datasets/model assets | Explicit dataset/model selection | Dataset registry and optional model cache | Reviewed licensed datasets/assets | Explicit download, then local | No startup training or surprise large downloads; preserve provenance/license/checksum. |

## Acceptance examples from the original prompt

| User request | Current bounded behavior | Evidence needed before accepting the full request |
|---|---|---|
| “Find where this image was taken.” | Embedded GPS and unverified local visual clues | Candidate map plus cited corroboration and contradictory clues; return unknown when insufficient. |
| “Find the earliest version of this image online.” | Not an integrated workflow | Actual matched pages, defensible publication evidence and an earliest-accessible timeline. |
| “Find visually similar images.” | Not an integrated workflow | Local or explicitly requested public candidate grid evaluated on crops/resizes/lookalikes. |
| “Research the public presence associated with this image.” | Non-biometric image clues can be inspected; search is separate | Object/logo/text/source research connected to citations; no face-to-private-person/account matching. |
| “Find this person across the public web where appropriate.” | Text search from supplied name/handle is possible | Public profile URLs and supporting/contradictory non-biometric evidence; no fabricated match. |
| “Find this object in these 50 videos.” | No end-to-end workflow | Bounded/resumable local indexing, reference-object matches, timestamps and confirmation of lookalikes. |
| “Find where this video originated.” | No end-to-end workflow | Timestamped keyframe/audio clues, actual matching uploads and separate origin/creator claims. |
| “Find where this video was filmed.” | No end-to-end workflow | Scene-specific candidate places with local observations, public corroboration and uncertainty. |
| “Compare these recordings.” | Optional STT adapters only | Local waveform/transcript comparison and cited segments. |
| “Find whether the same authorized speaker occurs.” | No end-to-end workflow | Consent-based probabilistic comparison with uncertainty and supporting sections, no identity claim. |
| “Search my computer for this object.” | Filename/file inspection is narrower | Granted local-media/document index and ranked occurrence evidence. |
| “Search the web for this object.” | Text web search available; image-to-query not composed | Reviewable visual clue query, source cards and actual object relevance checks. |
| “Analyze these satellite images.” | Generic image descriptions do not satisfy this | Georeferenced, date-aware aligned comparison with a validated method and source metadata. |
| “Clean this dataset.” | Trims strings, drops blank rows/exact duplicates and saves a separate artifact | Current bounded route has preservation/round-trip tests; richer cleaning requires explicit operation semantics and additional validation. |
| “Research this topic and build a report.” | Bounded research now saves an attributed Markdown report and cognition-compatible JSON evidence graph | Current report preserves observed excerpts, URLs, timestamp labels, failures and limits. Synthesized verified findings and an interactive graph remain separate work. |
| “Open Chrome and complete this workflow.” | Browser launch/URL routing verified | Observed page actions, intermediate checks, final-state verification and recovery. |
| “Find every public source related to this media.” | No exhaustive search guarantee | Bounded source discovery with searched scope/limits; never claim global completeness. |

Additional examples in the body of the prompt remain covered by the matrices:
open a project (desktop/files), scrape a site (web), research a building (images
and sources), track a vehicle across files (video/object locator), compare visible
appearance (bounded visual comparison), and perform a computer task (verified
observe/action loop). They are not separate product modes.

## Implementation order: complete vertical slices

1. **Local and polished dock baseline (P0).** Enforce local-only inference even
   with stale cloud settings; hide implementation names from ordinary UI; finish
   conversational rendering, meaningful working states, focus/history/settings,
   attachment handling and startup. Acceptance includes an actual packaged local
   conversation and image question with cloud AI network access unavailable.
2. **Documents and data to useful artifacts (P1).** Attach a PDF/DOCX/CSV/JSON or
   SQLite file, inspect it, ask a grounded local question, transform supported data
   and save a new artifact. Verify source unchanged and unambiguous failures for
   unsupported requests. This fills a large gap using existing code immediately.
3. **Research to evidence and report (P1).** Reuse `app/research` and
   `cognition.EvidenceGraph`; persist sources/claims/relations under the task and
   present source cards plus report export. Add supplied-name/handle public-profile
   queries without biometric matching. Prove partial-source failure preserves work.
4. **Local video and audio inspection (P1/P2).** Wrap real OpenCV frame/metadata
   methods and installed-only transcription in existing workers. Present keyframes
   and timestamped transcript before adding tracking or live microphone complexity.
5. **Image/object investigation (P2).** Add local detection/embeddings, overlays,
   similarity candidates and cross-file occurrence search. Compose explicit public
   clue/reverse-search requests and source timelines through the same evidence flow.
6. **Task composition, voice and verified actions (P2/P3).** Extend the persistent
   task system with dependencies; integrate local speech sessions and an observed
   browser/computer adapter. Every consequential action has a target, permission,
   outcome observation and recovery behavior.
7. **Maps and Earth observation (P3).** Add one complete licensed map/geocoder and
   dated satellite comparison workflow. Connect media location hypotheses to this
   same artifact/evidence system; do not substitute generic image captions for
   geospatial analysis.

Each slice ships through the dock with real input/output, visible work, cancellable
tasks, meaningful error states and an installer smoke test. It is not complete
when only a registry row, model download or standalone demo exists.

## Release checks and audit findings

The audit exercised `tests/test_browser_commands.py`, `test_desktop_tasks.py`,
`test_desktop_images.py`, `test_search_readiness.py` and `test_research.py` together;
the run passed with one platform-dependent skip. Those tests verify their bounded
contracts, not the unimplemented workflows above. The new UI/data/local-only
changes have additional integration coverage. On 2026-10-10 the final dock,
presentation, routing, queue and task run passed 116 checks with one platform skip;
the completed-response restoration check added afterward also passed. Visual
inspection covered the default 720 × 620 native dock and Settings. This does not
verify every DPI or multi-monitor arrangement. The packaged smoke test remains
required after building the intended committed sources.

Specific reuse hazards discovered in actual source:

- `sweep_core/integrations/audio.py` describes offline transcription but its quick
  and accelerated model helpers may download on first use; adapt explicit local
  paths before desktop exposure. The Vosk helper currently discards word timing.
- `sweep_core/integrations/vision.py::_download` uses an unverified TLS context.
  Do not connect this downloader or its face-research path to the desktop unchanged.
- `companion/tools/data.py::run_data_analysis` profiles, optionally groups and
  charts; it does not implement the cleaning promised by its docstring.
- `OpenCVEngine.extract_frames` returns image arrays without timestamps and reads
  through video frames. A desktop wrapper needs bounded seeking/work and timestamp
  artifacts before large-video use.
- `sweep_cognitive/perception/document.py` is representation/structure processing;
  it is not proof of native PDF/DOCX extraction readiness.
- The `app` research store is in memory, while desktop task JSON is persistent.
  A result export alone does not provide resumable research or a durable DAG.
- Existing model/registry license strings are inventory metadata. Code, weights,
  dataset terms and distribution obligations need separate review when adopting an
  optional capability.

Before replacing the installer: test local chat/image/file/data behavior with no
remote AI fallback; inspect actual native visuals; run task/permission/security
regressions; build from the intended committed sources; smoke-test outside the
repository; verify payload/checksums/upgrade preservation; then remove only the
superseded generated installer/build artifacts. Record remaining product gaps
honestly. The full original specification is larger than the current application.
