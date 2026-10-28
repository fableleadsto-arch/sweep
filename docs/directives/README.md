# Standing Sweep directives

Adopted on 2026-10-28 at the user's explicit request. Before every new Sweep
task, read **both full directives**, then the current development state:

1. [Enterprise master directive](SWEEP_ENTERPRISE_MASTER.md).
2. [Implementation resource catalog](SWEEP_IMPLEMENTATION_RESOURCES.md).
3. [Continuation state](../DEVELOPMENT_STATE.md).

The two directive files preserve the supplied attachment bytes. `manifest.json`
records their SHA-256 digests and attachment identifiers. Do not silently rewrite
the source directives to match implementation progress; record decisions, measured
results, departures and outstanding requirements in the audit/roadmap instead.

## How this changes the previous plan

- Preserve the Python native desktop companion and its installed functionality.
- Audit existing architecture before implementing the enterprise foundations;
  reuse current modules. TradeOps is the first commercial workflow. Vision,
  TestLab and Evidence are optional modules with separate business logic.
- The new commercial sequence supersedes the earlier Industrial/Construction
  expansion order. Earlier companion requirements remain a compatibility backlog.
- Local processing remains the current default. The new directive permits future
  opt-in cloud routing with permissions and cost settings; this is not permission
  to upload documents, enable a cloud backend or incur charges now. Shared compute
  is optional and must not be required for standalone operation.
- The resource catalog's title does **not** certify verification. Every candidate
  requires official-source/access/license/maintenance checks before adoption.
  Download only justified, permitted resources; keep measured and unmeasured
  evaluations clearly distinguished. No license or access right is inferred from
  the supplied list.
- Explicit authorization is needed before external messages, submissions, business
  record changes or irreversible actions. Preserve original evidence and user data.
- Proposed 95% field accuracy and 98% high-severity precision are targets, not
  achieved results. Do not claim enterprise readiness, demand or certification.

## Task preflight and completion

State which phase and acceptance criterion the next change serves. Inspect the
affected existing implementation and permission/data boundaries before editing.
Keep changes testable and focused. Record validation, limitations and the next
step in `DEVELOPMENT_STATE.md`; keep the architecture audit, roadmap and feature
matrix current. Commit only owned changes and preserve unrelated user work.

If a later user instruction changes scope, record it explicitly here or in the
roadmap rather than silently discarding these directives. Instructions with
higher priority and later explicit user instructions continue to take precedence.
