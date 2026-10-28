# Sweep continuation state

Updated: 2026-10-28. Read this file when the user says **continue**. Keep one
current backlog here; update evidence and the next action after each milestone.
The full original requirement mapping remains in `PRODUCT_REQUIREMENTS.md`.

## Product contract

Python-based installed desktop companion, one native top-screen chat dock with
history and settings. Users describe tasks or converse in the same chat. Local
AI inference remains the default; public web retrieval is a separate, requested
tool. The Oct28 directive permits future opt-in cloud processing with permissions
and cost limits; no cloud upload, charge or backend activation is authorized by
that architectural option alone.
Hide model/vendor details from routine interaction, retain honest diagnostics.
Preserve personal data and unrelated development work. Test focused changes,
commit and push them, then validate and replace the installer. Do not equate a
registered tool, UI control, or passing mock test with an operational feature.

## Latest direction and order

Before **every new Sweep task**, read both full versioned directives:
[enterprise master](directives/SWEEP_ENTERPRISE_MASTER.md) and
[implementation resources](directives/SWEEP_IMPLEMENTATION_RESOURCES.md), then
this file. They were adopted Oct28 and copied byte-for-byte from the attachments;
`directives/manifest.json` preserves their hashes. [Adoption rules](directives/README.md)
explain precedence and local/cloud permissions.

The current commercial order is **audit -> shared foundations -> TradeOps -> pilot
-> commercial readiness -> optional Vision/TestLab/Evidence**, preserving the
general-purpose Python desktop companion. This supersedes the old Industrial /
Construction expansion order. The [implementation roadmap](IMPLEMENTATION_ROADMAP.md)
tracks phase gates and all nine required deliverables; the
[architecture audit](ENTERPRISE_ARCHITECTURE_AUDIT.md) contains the feature matrix.

Phase 0 is active, not complete. A structural pass parsed 729 first-party Python
files with zero syntax errors; runtime boundaries and existing source/release
evidence were reviewed. All 43 resource candidates are tracked; four official
repositories were checked initially, without adopting dependencies or downloading
datasets. Rights, maintenance, compatibility and performance reviews remain open.
See [the registry](SOURCE_AND_DATASET_REGISTRY.md).

Adoption validation: both repository directive copies match the supplied attachment
bytes and SHA-256 hashes; 28 local documentation links resolve; registry IDs cover
all 43 candidates exactly once. Git attributes preserve the supplied line endings
so the manifest remains valid after checkout. The AGENTS.md change is limited to
the mandatory-read instruction; the user's separate dataset catalog remains untouched.

Companion compatibility priorities remain: cold reply latency, dependable device
actions, same-chat continuity, native UI usability, bounded memory, real public
research coverage and explicit failure recovery. Their verified improvements and
limits are below. Registries do not establish 120 working tools or general intelligence.

Optional shared compute must preserve standalone use and keep secondary devices
lightweight. Existing authenticated text hosting remains source-only with native
pairing, multi-device TLS, expiry/crash recovery and broader tenant isolation still
missing. Never send confidential business files to untrusted peers. Do not expand
hosting ahead of the first coherent TradeOps workflow.

## Current verified release

Installed code release: `3ae1258be` (native composer and explicit dependency
recovery), committed and pushed. Installer: `dist/Sweep-Setup.exe`, 126,040,375
bytes. Its normal upgrade completed Oct28 and preserved 44 personal files
hash-for-hash. Source tree dirty provenance is recorded; unrelated work remains
unstaged. Later directive/audit documentation commits do not change this binary.

Packaged and installed verification passed app/setup checksums, all 3,110 installed
payload CRCs, 27 native-library hashes and 28 license-notice hashes. Native installer,
packaged dock and installed dock launch/clean-exit checks passed. The dock smoke
used synthetic interrupted-task history. Installer remains unsigned.

Seven real packaged workers outside the repository passed: calculator 1.70 s,
data 1.35 s, text 0.78 s, PDF 1.12 s, DOCX 0.65 s, OCR 2.02 s and local chat
24.14 s (five preview events). Small synthetic fixtures do not establish broad
accuracy or percentile latency. The recovery UI driving packaged workers completed
a restarted CSV-cleanup -> JSON chain in 3.99 s; with installed workers it took
3.14 s. Both required two approvals, returned two rows and preserved original
CSV bytes and old task references. No claim that cold inference latency is fixed.

Previous installed-release app launch evidence (Oct22): Character Map opened an
independently observed window in 3.90 seconds. Its shorter product verification
window reported only accepted execution. Sampled video previously took 111.63
seconds. Those paths were not re-benchmarked in this evidence-storage release.

## Source prompts

- Original desktop prompt: local attachment
  `d57a0fcc-302b-4358-9d23-1b3149d06156/Pasted text.txt`.
- New multi-industry prompt, read 2026-10-14: local attachment
  `619db2be-9d61-471a-a329-3aec00989c0d/Pasted text.txt`.
- Latest explicit user corrections take precedence: a usable local companion,
  reduced resource use, reliable device actions and remembered continuation work.
- Both supplied prompts were reread on 2026-10-23. The latest direction is to
  improve the already-built infrastructure at that milestone. Prioritize reliable companion tasks,
  recovery and evidence, then the first TradeOps workflow. The optional text host
  is useful existing work, but does not satisfy those desktop requirements.
- Oct28 enterprise master and resource directives supersede the commercial
  sequence above; their full repository copies are linked at the top of this file.
  Both were read in full and preserved with attachment hashes. Phase 0 remains active.

## Repair evidence (2026-10-21)

- `6f8a70ce4`: conversation-scoped document/data follow-ups and this continuation
  file; `b1688cffb`: retain completed native cards during streaming.
- `48c4b74ee`: native local chat stream/resource bounds; `2436c7d4d`: installed-app
  discovery/launch receipts; `bf1d9a214`: mesh integrity and bounded retention.
- Source native dock + real local model compared synthetic invoice quantity 12
  against packing-list quantity 10 in 15.79 seconds. Queued deduplication then JSON
  conversion took 2.41 seconds and preserved the original CSV.
- Real Character Map launch: observed its owned window in 2.12 seconds; only the
  test-owned window was closed afterward.
- Oct18 chat measurements: cold greeting first text 14.14 s / total 16.63 s;
  warm companion first text 2.69 s / total 10.05 s; warm supplied-name/project
  recall 4.66 s / 6.86 s. A repeated cached-prefix trial was faster, but is not a
  general response-time guarantee. Model allocation was about 2.16 GB on CPU.
- Focused checks: 302 passed across providers/streaming/apps/controller/mesh;
  118 mesh checks passed after adding retention bounds; 58 context/follow-up checks
  passed with one Windows symlink-fixture skip. Earlier desktop integration set:
  242 passed, two symlink skips. Broad release regression: **933 passed, six
  skipped** (Windows symlink fixture privileges), 153.15 seconds.
- Known gap: source labels and reviewable excerpts are deterministic; model-written
  inline citations and factual interpretations still require checking. This is
  not the complete TradeOps reconciliation workflow.

## Shared desktop evidence milestone (2026-10-23)

- Reused the existing task store and context result checks in one lightweight
  `desktop.results` contract. Completion, native chat/history, result actions,
  exports and follow-up sources all use it. Reads reject redirected/nonregular,
  changed, oversized or damaged files; finite JSON, unique keys, bounded nesting
  and the fields used by native views are validated. Result bytes are capped at
  2 MB, nesting at 64 levels and structure at 100,000 values. These are application
  bounds, not OS isolation, tamper-proof evidence or total process-memory limits.
- Completion saves validated UTF-8 evidence atomically before updating status.
  Disk/validation failures fail the active task and allow the queue to continue.
  A crash between evidence and metadata writes remains interrupted on recovery.
  Earlier records/results are retained; retry creates a fresh task and approval.
- Damaged/missing results show a native recovery card and are excluded from chat
  context and export. Actions reject another conversation's task, including an
  inconsistent chat record. Existing seven completed personal results all passed
  the new reader without modification. The recovery card was visually inspected.
- A delayed transcript scroll callback now has its Qt owner as context, so closing
  the transcript cancels the callback instead of accessing a destroyed widget.
- Native source dock with real local workers/model: synthetic invoice quantity
  12 vs packing quantity 10 comparison completed in 36.28 s (31 preview events);
  queued deduplication-to-JSON finished in 1.95 s and preserved the source CSV.
  This run overlapped regression work and does not establish improved latency.
- Full desktop/dock regression: **645 passed, six skipped**, 94.30 s. Skips are
  Windows symlink-fixture privileges. Includes actual worker queue failure/retry,
  corrupted/missing/oversized result recovery, cross-chat action denial, interrupted
  persistence and transcript destruction. Ruff and owned-file whitespace checks
  passed. Native packaging and upgrade completed; see the current release above.
- Still missing: explicit dependency retry/resume, project workspaces, complete
  TradeOps extraction/reconciliation, broad operational OSINT coverage and mesh
  integration. Cold inference startup remains slow. Do not mark the main prompt
  complete based on this infrastructure repair.

## Native recovery and UI milestone (2026-10-24)

- The dock groups attachment, input and send controls into one composer; input
  grows for up to five explicit lines. Empty Send is disabled, Stop appears only
  for active work, and the conversation title/queue summary remain visible.
  The compact header and secondary-action menu keep narrow layouts usable.
- Failed/interrupted follow-ups offer **Review recovery**. The panel retains
  valid original sources and offers completed retry descendants from this chat,
  showing a saved excerpt. Replacements must be selected explicitly. **Retry
  source** creates one task through existing approval/worker paths; return to
  the panel after completion to retry the follow-up with new source IDs.
- Selection is revalidated after the modal review. Other chats, changed request
  scopes, running/queued results, damaged evidence and cyclic retry chains are
  excluded. Up to eight available choices per source are read, including a valid
  original and the most recent retries. Records/grants remain immutable; normal
  remembered permissions still apply. Cancelling review or denying an action
  creates no work. No automatic computer-action replay or implicit fallback.
- Source dock fresh-start check recovered a queued follow-up as interrupted,
  then retried a failed CSV cleanup and explicitly rebound JSON conversion:
  two approvals, two output rows, original CSV/old references preserved, 1.34 s.
  Welcome, 420-pixel dock, recovery review and recovered artifact were inspected.
- UI checks found and fixed stale deferred scroll restores. One Qt-owned timer
  now coalesces updates, preserving selections/reader position and cancelling
  safely on widget destruction. No idle animation/timer was added.
- Broad desktop/dock regression: **656 passed, six Windows symlink-fixture
  skips**, 78.68 s. Final recovery/chat checks: **46 passed**; final bounded-choice
  checks: **eight passed**. Ruff passed. Native packaging, installed upgrade and recovery checks
  completed Oct28; see the current release above.
- Limits: manual recovery one dependency at a time, bounded to retained chat
  records; no general DAG or project workspace. A missing original task record
  requires reattaching/recreating the request. Choosing a saved result cannot
  restore a deleted generated artifact. TradeOps and cold latency remain open.

## Next action and release gate

Next: finish the Phase 0 TradeOps product contract, focused threat model,
evaluation protocol and resource/cost baseline. Then add source-linked structured
document ingestion using the current local adapters, followed by workspace and
correction state, deterministic invoice/packing-list reconciliation and native
side-by-side review. Follow the numbered increments in IMPLEMENTATION_ROADMAP.md.
Generic model comparison is not TradeOps field extraction. Do not claim the new
master prompt, pilot targets or all resource checks are complete.

Keep the companion operational and test native progress, long/narrow layouts,
source navigation, permissions, failure recovery and original-file preservation.
Document-only continuation updates do not need a new installer. Runtime changes
must pass packaged and installed checks before replacement.

Hosting backlog remains: native pairing/host settings and a lightweight client
with explicit certificate trust and OS-protected credentials. Route only text
after consent; keep desktop actions and local file grants on the client. Local
standalone remains the default; a paired second device must not need a model.
Use `companion.hosting`, not the legacy global token/`RemoteWorker` example.

For that host, validate real multi-device TLS/load behavior, timed result retention and
crash recovery. Broader remote files/tasks need tenant workspaces and OS isolation
before exposure. Cold local reply latency is still a priority: the latest real
host run took 32.55 seconds for its first reply; shared queuing does not fix cold
model loading. Do not describe the companion as responsive in all conditions.

The legacy capability regression also exposed unavailable symbolic math, charts,
graphs, vision and NLP dependencies in the development environment; include them
in the operational-tool audit, not the count of working OSINT features.

Complete targeted code fixes and meaningful tests; record actual latency/app
launch measurements; run relevant regression checks; commit only owned changes;
push; rebuild the native installer; exercise the packaged paths outside the repo;
upgrade and verify preservation. Record remaining failures rather than claiming
the complete master prompt is implemented.

## Shared compute foundation evidence (2026-10-22)

- New host pool defaults: one running process, 32 outstanding jobs globally,
  eight per owner, 64 KiB JSON job specifications, 64 completed statuses in memory,
  and a five-minute timeout. Queue limits include running work. Dispatch is fair
  by job, not by equal CPU time; it does not balance multiple machines.
- Status and cancellation are owner-scoped; IDs are host-generated. The trusted
  caller must supply authenticated identity. This is not authentication or an OS
  sandbox; worker processes still run under the host account.
- Cancellation waits for actual process exit before slot reuse. Crashes without
  valid results fail honestly; unsafe IDs, duplicate overwrites and invented
  cancellation records are rejected. Child status writes use atomic replacement
  and bounded Windows file-lock retries. Task kinds no longer silently fall back.
- Removed eager NumPy import; CPU version/probes and pool import require no
  numerical/model frameworks. CPU availability/support now reflect dependencies
  and operations it actually implements. Idle pool threads exit automatically.
- **83 passed** across pool, existing compute and API checks (19.81 seconds),
  including two real owner-scoped CPU jobs, actual process crash/cancel tests,
  concurrent admission and an isolated import check. One upstream test-client
  deprecation warning remains. Ruff and owned-file whitespace checks passed.
- Wider legacy capability run: 94 passed, five failed and three skipped. The five
  failures are missing optional dependencies (`sympy`, `matplotlib`, `networkx`,
  `cv2`, `spacy`/`nltk`), independently confirmed absent. This is not an all-suite
  pass. No optional packages were added to the lightweight desktop build.
- Pool state is in memory; persisted worker files have no retention/restart
  recovery policy yet. This source-only host milestone does not replace installed
  desktop code release `4d42bba69` or claim a working network-hosting product.

## Authenticated text host evidence (2026-10-22)

- Source milestone only: `companion.hosting` CLI creates organizations/users and
  ten-minute single-use invitations locally. Device credentials expire after
  30 days; only high-entropy secret digests are stored in SQLite. Every API call
  rechecks credentials; multiple devices share their user's queue quota.
- Separate native-client API supports pairing, own identity/revocation, text jobs,
  bounded status previews/results and cancellation. No legacy execution routes,
  model selection, client-provided identity, file access or computer-action routes.
- Loopback-only HTTP; non-loopback CLI binding requires TLS and explicit host
  names. Forwarded headers are not trusted. Browser-origin requests, query tokens,
  oversized bodies and extra execution fields are rejected; validation errors
  do not echo credentials. No network service was left running during this work.
- Text inference reuses downloaded-model validation and local streaming in real
  child processes. Normal terminal/shutdown paths delete host-owned temporary
  prompt/status files; personal configuration and history remain untouched.
- **154 tests passed** (37.20 s): identity/API/worker/CLI, existing pool/compute/API.
  Includes concurrent invite redemption, expiry, revocation, user boundaries,
  quotas, transport/body limits, controlled-engine subprocesses, cancellation and
  cloud-alias rejection. One upstream test-client deprecation warning remains.
  Ruff checks passed. Earlier optional capability dependency gaps remain open.
- Actual local TCP host + installed model, two synthetic users: first preview
  31.59 s, first completion 32.55 s; second completion 35.31 s including waiting.
  Second job was observed queued. Foreign read/cancel denied, revoked credential
  rejected, all job files/processes cleaned and personal settings hash preserved.
- Remaining: native client pairing/trust/protected token storage; multi-machine
  TLS/load tests; idle result expiry and crash residue cleanup/recovery. Revocation
  blocks new API requests but does not cancel already admitted work. Host operator
  can access host-side data; owner checks are not OS isolation. Installer still
  identifies desktop code `4d42bba69`; no desktop behavior changed this milestone.
