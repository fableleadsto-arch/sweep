# Sweep continuation state

Updated: 2026-10-18. Read this file when the user says **continue**. Keep one
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

1. **Chat latency and companion behavior — active.** Measure time to first visible
   text and completion on this machine. Fix streaming, role/context handling and
   unnecessary work. Preserve local-only execution and cancellation. Measure cold
   and warm behavior separately; do not claim an improvement without evidence.
2. **Device actions — active.** Resolve installed applications, understand natural
   launch requests, report real launch failures and verify supported actions.
   Preserve browser intent such as “open YouTube in Brave.”
3. **Conversation continuity — active.** Complete the existing uncommitted work:
   follow-ups reference saved results in the same chat; compare up to four
   documents; chained data transformations consume the earlier output. Test
   failure/retry, source isolation and native dock integration.
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

## Next release gate

Complete targeted code fixes and meaningful tests; record actual latency/app
launch measurements; run relevant regression checks; commit only owned changes;
push; rebuild the native installer; exercise the packaged paths outside the repo;
upgrade and verify preservation. Record remaining failures rather than claiming
the complete master prompt is implemented.
