# Enterprise threat model — initial local workflow

2026-10-28. Status: **Partially implemented**. This is a design review and test
plan, not a penetration-test report or compliance certification.

## Assets, actors and boundaries

Protect original documents, extracted business data, personal chat/task history,
credentials, corrections, reports and action approvals. Treat document authors,
retrieved pages, model outputs, remote clients and untrusted peers as potentially
hostile. A local process running as the same OS user can read/modify ordinary
plaintext storage; application hashes cannot protect against that attacker.

Native chat -> explicit file grant -> bounded worker -> extraction snapshot ->
workspace-authorized review -> separately approved export/external action. Every
arrow requires validation. An attachment is observation input, never executable
instruction or a credential. The current additive snapshot API is not an access
control service. The desktop's same-chat controls and optional host's identities
do not establish enterprise workspace/tenant isolation.

| Threat | Current boundary / limitation | Required evidence before expansion |
| --- | --- | --- |
| Wrong or changed source/quote | Snapshot hashes and exact quote/range checks; hashes are not authenticity proof | Mutated/cross-document/parser-version/page-order reference tests; workspace lookup must precede resolution |
| Unapproved file reads | Existing exact local file grants and bounded reader | Deny directory/foreign grants, network shares, redirected paths and changed-file races in the eventual workspace ingestion path |
| Parser resource exhaustion | Existing 20 MB input, PDF decompression/page/text limits; subprocess timeout | Oversized/compressed/malformed fixtures; OS memory/CPU policy remains missing |
| Scans or partial extraction shown as complete | Explicit truncation/no-text status in source contract | Blank/mixed/scanned/truncated-page fixtures; no automatic missing-field verdict from partial extraction |
| Prompt injection | No model/tools in snapshot API; external text remains data | Adversarial document commands cannot authorize tools, override grants, send messages or change records |
| Cross-workspace/tenant leakage | No enterprise workspace service yet | Membership-scoped storage/query/export/delete tests before any shared documents; unpredictable IDs are insufficient |
| Path/action injection | Reference schema contains no file path or executable fields | Strict unknown-field rejection; source navigation resolves through trusted workspace lookup, never a document-provided path |
| Credential exposure / network exfiltration | Desktop AI local by default; stored Windows provider secrets use DPAPI | No model/network access in ingestion tests; future cloud permissions, cost limits and redacted logs |
| SSRF | Reviewed public-web gateway validates addresses/redirects; not every optional client covered | All new connectors use reviewed transport; test private/loopback/mixed DNS/redirect targets and credentials |
| Malicious business edits / duplicate sends | TradeOps live connectors not implemented | Exact reviewed action/version/destination, fresh approval, idempotency and ambiguous-outcome handling |
| CSV/Excel formula execution | Future report adapter | Dangerous leading characters/whitespace and hyperlink/formula payload tests; preserve raw value in safe JSON |
| Corrupt persistence / crash | Existing result-before-success storage and manual retry | Schema compatibility, interrupted writes, backup/restore, stale corrections and no automatic action replay |
| Malware / active content | Readers do not intentionally execute macros; no malware scanning integration | Scanner policy and explicit unavailable/failure handling before hosted uploads; subprocess is not an OS sandbox |
| Retention / deletion | Current personal local files persist | Workspace-scoped retention/export/delete including derived copies/backups; no silent original deletion |
| Shared compute abuse | Experimental text-only host has auth, queues and limits | Multi-device TLS, revocation of admitted work, TTL/crash cleanup and isolation before enterprise deployment |
| Supply-chain compromise | Current build checksums/provenance; unsigned Windows build | Dependency/model license and vulnerability review, signed update trust and rollback policy |

## Release boundaries

The first source-only library reads already granted files through existing readers;
it neither exposes an API server nor modifies originals, workspace records or
configuration. No migration is needed. Tests must demonstrate no GUI/model/network
imports during this path. Denial or parse failure returns an error; it must not
produce a successful but empty business result.

Before the desktop routes real TradeOps tasks, enforce task approvals and workspace
ownership, validate immutable snapshot serialization in the worker/result boundary,
and test original preservation and cancellation. Before hosted customer inputs,
complete tenant isolation, TLS, quotas, sandbox/scanner policy, secret protection,
retention and operator-access disclosure. No confidential inputs go to untrusted
peers. Proposed zero known cross-tenant exposures is a release stop condition,
not a claim that finite tests prove absence of vulnerabilities.
