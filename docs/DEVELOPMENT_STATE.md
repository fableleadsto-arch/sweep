# Sweep continuation state

Updated: 2026-10-21. Read this file when the user says **continue**. Keep one
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

1. **Chat latency and companion behavior — source validated; packaging pending.**
   Native chat roles replace flattened prompts; small response packets display
   immediately, with bounded 4K context, two-minute residency and one model/context
   when Sweep starts the engine. CPU thread trials did not justify hardcoding a
   thread count. Cold loading remains slow; further quality/latency work is needed.
2. **Device actions — source validated; packaging pending.** Installed apps resolve
   through aliases, App Paths, executable paths, Start Menu and registered Store
   entries. Missing/ambiguous apps do not silently become web searches. Native
   receipts distinguish accepted requests from observed windows. Browser intent
   such as “open YouTube in Brave” remains supported.
3. **Conversation continuity — source validated; packaging pending.** Saved results
   stay scoped to the same chat; up to four documents can be compared. Queued data
   follow-ups consume the earlier transformed output. Failures stop dependent
   work; source labels remain visible even if the model omits inline citations.
4. **Resource use and fair multi-user scheduling — audit active.** Bound local
   inference, queues, context and idle memory. The user clarified on 2026-10-18:
   start with a locally hosted installation; as more users join, share the heavy
   processing on a host while additional devices stay lightweight. Preserve a
   local standalone mode. Design an optional trusted host with authenticated
   user/organization isolation, bounded per-user queues, fair dispatch,
   cancellation and load validation. Desktop actions stay on the requesting
   device. Do not silently turn other users' computers into compute workers or
   share their files. Pairing, consent and routing need an explicit product flow.
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

Installed/source release: `8072f3c39` (media and local speech). Installer:
`dist/Sweep-Setup.exe`. Native install/launch/exit, real local transcription,
sampled video, OCR, documents, data and chat were exercised. Installed payload
CRC checks covered 3,110 files; upgrade preserved 26 personal files.

Known performance: packaged local greeting took 14.6 seconds in one run; sampled
video vision took 111.63 seconds. Those observations do **not** establish normal
or worst-case response time. Chat/app-launch/resource fixes are not yet shipped.

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
  242 passed, two symlink skips. A broader release regression run is pending.
- Known gap: source labels and reviewable excerpts are deterministic; model-written
  inline citations and factual interpretations still require checking. This is
  not the complete TradeOps reconciliation workflow.

## Next release gate

Complete targeted code fixes and meaningful tests; record actual latency/app
launch measurements; run relevant regression checks; commit only owned changes;
push; rebuild the native installer; exercise the packaged paths outside the repo;
upgrade and verify preservation. Record remaining failures rather than claiming
the complete master prompt is implemented.
