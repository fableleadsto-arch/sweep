# Sweep continuation state

Updated: 2026-10-22. Read this file when the user says **continue**. Keep one
current backlog here; update evidence and the next action after each milestone.
The full original requirement mapping remains in `PRODUCT_REQUIREMENTS.md`.

## Product contract

Python-based installed desktop companion, one native top-screen chat dock with
history and settings. Users describe tasks or converse in the same chat. Local
AI inference is required; public web retrieval is a separate, requested tool.
Hide model/vendor details from routine interaction, retain honest diagnostics.
Preserve personal data and unrelated development work. Test focused changes,
commit and push them, then validate and replace the installer. Do not equate a
registered tool, UI control, or passing mock test with an operational feature.

## Latest direction and order

The user reports minute-long replies, unreliable application launching, weak
companion behavior, missing OSINT tools, excessive resource use and incomplete
neural-mesh integration. Fix those failures before expanding the product.

1. **Chat latency and companion behavior — installed release validated.**
   Native chat roles replace flattened prompts; small response packets display
   immediately, with bounded 4K context, two-minute residency and one model/context
   when Sweep starts the engine. CPU thread trials did not justify hardcoding a
   thread count. Cold loading remains slow; further quality/latency work is needed.
2. **Device actions — installed release validated.** Installed apps resolve
   through aliases, App Paths, executable paths, Start Menu and registered Store
   entries. Missing/ambiguous apps do not silently become web searches. Native
   receipts distinguish accepted requests from observed windows. Browser intent
   such as “open YouTube in Brave” remains supported.
3. **Conversation continuity — installed release validated.** Saved results
   stay scoped to the same chat; up to four documents can be compared. Queued data
   follow-ups consume the earlier transformed output. Failures stop dependent
   work; source labels remain visible even if the model omits inline citations.
4. **Resource use and fair multi-user scheduling — experimental text host validated.** Bound local
   inference, queues, context and idle memory. The user clarified on 2026-10-18:
   start with a locally hosted installation; as more users join, share the heavy
   processing on a host while additional devices stay lightweight. Preserve a
   local standalone mode. Design an optional trusted host with authenticated
   user/organization isolation, bounded per-user queues, fair dispatch,
   cancellation and load validation. Desktop actions stay on the requesting
   device. Do not silently turn other users' computers into compute workers or
   share their files. Pairing, consent and routing need an explicit product flow.
   `FairComputePool` now supplies bounded owner-scoped dispatch over the existing
   real subprocess worker, with round-robin queues, cancellation and timeouts.
   A separate opt-in text host now adds persistent organization/user/device
   identities, single-use invitations, revocable credentials and a strict local
   inference contract. A temporary loopback host completed two actual-model jobs.
   It is not connected to the desktop or installed as a persistent service. See
   [SHARED_COMPUTE.md](SHARED_COMPUTE.md) for limits and remaining host requirements.
5. **OSINT and neural mesh — audit active.** Count actual executable tools and
   desktop routes, identify unavailable dependencies and placeholder results.
   Connect useful public-source research through existing permissions, evidence
   and progress contracts. Never promise “120+ working tools” from registry count.
   Mesh integrity repairs are source-tested: failure propagation, dependency skips,
   estimated-budget admission and truthful cache confidence/provenance. The mesh
   cache accepts at most 32 small supported values (1 MiB estimated Python size
   each), skipping opaque/tensor outputs before copying. Node/engine histories
   retain 64 recent executions by default, configurable down to zero. These are
   retention bounds, not OS enforcement of execution RAM or distributed scheduling.
6. **Shared workflow foundation.** Reuse existing tasks, approvals, document/data
   adapters and evidence records. Add project/workspace isolation, resumable
   dependencies and verified results where missing; avoid duplicate frameworks.
7. **First industry workflow: TradeOps.** Shipment document reconciliation:
   invoice, packing list, bill of lading and origin documents; extract fields,
   compare names/quantities/weights/values/currencies/references, flag missing and
   contradictory evidence, cite pages and produce a saved review package. Test
   realistic nonsensitive fixtures and integrate with the same chat. No claim of
   final customs/legal classification or autonomous official submission.
8. **Then Industrial**, narrow component/specification and supplier research;
   **then Construction**, evidence timelines and contract-linked delay review.
9. **Later modules**, each independently validated: environmental compliance,
   insurance, supplier risk, property, infrastructure, regulations, and healthcare
   administration only after its data/security requirements are satisfied.
10. **Remaining original companion scope:** stronger image/object investigation,
    research chaining, evidence visualization, supported desktop/browser
    observe-act-verify loops, live voice, richer document/tabular formats and
    media workflows. See the full requirement audit for actual limits.

## Current verified release

Installed code release: `4d42bba69` (companion repairs), committed and pushed.
Installer: `dist/Sweep-Setup.exe`, 126,022,215 bytes. Native installer and installed
dock launch/exit passed. Oct22 verification checked SHA256SUMS, 27 native library
hashes, 28 license notice hashes and all 3,110 installed payload CRCs. The upgrade
preserved 44 personal files hash-for-hash. Unrelated user work remains unstaged;
the build metadata correctly records a dirty source tree.

Packaged workers ran outside the repository: calculator, data, text, PDF, DOCX,
OCR and local chat passed; greeting took 11.48 seconds with five preview events.
The native dock driving packaged workers compared invoice quantity 12 with packing
quantity 10 (28.98 seconds under build load), then completed queued CSV cleanup
and JSON conversion (2.53 seconds), preserving the original file. Its expanded
conversation and source controls were visually inspected.

Installed Character Map launching opened an independently observed window in
3.90 seconds. The product's shorter verification window reported only that
Windows accepted the request, correctly avoiding a false verified-success claim.
Cold chat still takes seconds; sampled video took 111.63 seconds in the previous
media release. These observations are not normal/worst-case time guarantees.

## Source prompts

- Original desktop prompt: local attachment
  `d57a0fcc-302b-4358-9d23-1b3149d06156/Pasted text.txt`.
- New multi-industry prompt, read 2026-10-14: local attachment
  `619db2be-9d61-471a-a329-3aec00989c0d/Pasted text.txt`.
- Latest explicit user corrections take precedence: a usable local companion,
  reduced resource use, reliable device actions and remembered continuation work.
- The user will provide a revised main prompt later. Keep the current industry
  order recorded, but prioritize the companion repairs and hosting foundation
  while that later specification is pending.

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

## Next action and release gate

Next: implement native desktop pairing/host settings and a lightweight client
with explicit certificate trust and OS-protected credentials. Route only text
conversation to a paired host after consent; leave local file grants and approved
desktop actions on the client. Standalone mode remains the default, and paired
mode must not require downloading a model on the second device. Use the new
`companion.hosting` endpoints, not the legacy global token/`RemoteWorker` example.

Then validate real multi-device TLS/load behavior, timed result retention and
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
