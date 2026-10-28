# Sweep implementation roadmap

Adopted 2026-10-28. Read both [standing directives](directives/README.md) in full
before every new task, then [DEVELOPMENT_STATE.md](DEVELOPMENT_STATE.md). This is
the current execution order; the original companion backlog remains supported.
Statuses use the master directive's vocabulary. Targets are not achievements.

## Phase gates

| Phase | Status | Work and acceptance evidence |
| --- | --- | --- |
| 0: Audit and validation | Partially implemented | Structural inventory and initial architecture audit recorded; registry started. Finish precise TradeOps spec, threat model, evaluation/cost baseline and dependency assessment. Customer access is currently unavailable. |
| 1: Shared foundations | Partially implemented | Existing native tasks/results/approvals/local adapters reused. Add versioned source-linked document contracts, workspace ownership, correction/audit storage, ingestion failure handling and meaningful boundary tests. |
| 2: TradeOps MVP | Planned | Invoice + packing-list workflow; deterministic discrepancies, native side-by-side source review, corrections, CSV/Excel report, draft communication and separately approved follow-up. |
| 3: Pilot readiness | Planned | Representative authorized corpus, measured exact-match/alert precision and sample sizes, resource/cost metrics, security tests, onboarding, deployment/recovery. |
| 4: Commercial readiness | Planned | Metering/billing experiments, team roles, monitoring/support, signed reliable updates and measured capacity. No hardcoded pricing before validation. |
| 5: Optional modules | Planned | Independently enableable Vision, TestLab and Evidence; validated separately. Small synthetic fixtures support internal QA earlier. |

## Next increments in order

1. **Finish the bounded Phase 0 design.** Specify an initial workflow: one commercial
   invoice and one packing list for the same shipment, local text-bearing PDFs
   first. Record unsupported scanned/layout cases explicitly; OCR/table support
   follows measured parser selection. Define field/source schemas, matching rules,
   security boundaries and a synthetic gold fixture suite. Do not fabricate fields
   or pretend arbitrary layouts are supported by template-specific extraction.
2. **Implement source-linked ingestion.** Preserve original SHA-256, page text and
   bounded source locations using `desktop.data_tools`; distinguish unavailable
   regions from exact coordinates. Schema versions, missing/ambiguous states,
   parser/version metadata and deterministic tests. No extra model download needed
   for text extraction. Add migration compatibility checks for existing results.
3. **Implement workspace and review state.** Explicit local workspace IDs/membership,
   document versions, corrections retaining original values, append-only application
   audit events and export/delete behavior. Backup/rollback and cross-workspace tests
   precede data migration. Do not describe application logs as tamper-proof.
4. **Implement TradeOps rules and chat route.** Validate quantities/units, identifiers,
   currencies, totals, dates, duplicates and required fields. Use decimal arithmetic;
   avoid fuzzy matching silently assigning unrelated lines. A discrepancy links to
   both source fields and states its rule and uncertainty. No LLM is needed for
   arithmetic. Approvals, cancellation and retry reuse current task infrastructure.
5. **Make the native review workflow usable.** Side-by-side documents, discrepancy
   list, clickable page/region evidence, correction history and reviewable report.
   Verify narrow dock/expanded workspace, keyboard navigation, long files, failures
   and interrupted jobs. Controls ship only with actual behavior.
6. **Complete report and follow-up.** CSV and Excel exports with formula-injection
   protection, original files preserved; draft email is a saved artifact. Sending,
   portal submissions and modifying a business record require separate approval of
   the exact action, destination and data. No live connector is assumed available.
7. **Measure and prepare pilot.** Use permitted representative data and reproducible
   extraction/reconciliation scoring. Record cold/warm timings, total process RAM,
   per-document cost and reviewer correction time. Proposed 95% critical-field
   exact match and 98% severe-alert precision require actual denominators and
   uncertainty intervals. Stop release on any known cross-tenant exposure.

## Required deliverables and current state

| Deliverable | Status | Completion criterion |
| --- | --- | --- |
| [ENTERPRISE_ARCHITECTURE_AUDIT.md](ENTERPRISE_ARCHITECTURE_AUDIT.md) | Partially implemented | Initial inventory/map/risks measured; deeper dependency/security and footprint assessment remains. |
| [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) | Partially implemented | Governing sequence and acceptance gates recorded; refine estimates and implementation scope as Phase 0 closes. |
| [SOURCE_AND_DATASET_REGISTRY.md](SOURCE_AND_DATASET_REGISTRY.md) | Partially implemented | All 43 candidates tracked; verify relevant resources before use, then complete remaining reviews by module. |
| `TRADEOPS_PRODUCT_SPEC.md` | Planned | Precise initial document/field schemas, matching, correction and UI acceptance examples. |
| `SECURITY_THREAT_MODEL.md` | Planned | Assets/actors/boundaries/mitigations/tests for local and optional shared workflows; implemented vs missing controls. |
| `EVALUATION_PROTOCOL.md` | Planned | Rights-cleared fixtures, split policy, labels, deterministic scoring, failure strata, sample size and confidence reporting. |
| `MARKET_VALIDATION.md` | Planned | Customer hypotheses, interview protocol and actual findings; no invented interviews or demand. |
| `COST_AND_UNIT_ECONOMICS.md` | Planned | Instrumentation plan and measured local/cloud/support costs; unmeasured quantities explicit. |
| `DEPLOYMENT_GUIDE.md` | Planned | Existing desktop instructions extended with module setup, backup/restore, update/rollback and security configuration. |

Existing companion latency/device/recovery regressions remain release gates. Do
not replace the working native shell, erase personal history or expand optional
model dependencies to tick a feature box. Continue public-source tool coverage
and lightweight host work only where it does not displace the first coherent
TradeOps workflow. Unavailable pilot data blocks customer claims, not local
implementation and synthetic QA.

## Completion protocol

For each increment: state its acceptance criterion, inspect existing modules,
implement a focused change, test real failure boundaries and relevant regression,
record measured results/remaining gaps, commit and push owned changes. For runtime
changes rebuild, exercise packaged behavior outside the repository, verify
upgrade/personal-data preservation and update the installer. Documentation-only
commits do not require a new binary or rewriting its source revision.
