# Deployment and verification

2026-10-28. Status: **Partially implemented**. Installed product: Windows native
companion. Enterprise foundation: source-only Python API; TradeOps UI/workspaces
are not installed features yet. No hosted enterprise deployment is certified.

## Existing desktop

Follow [DESKTOP.md](DESKTOP.md) for per-user setup, native dock, tray, shortcut,
local model configuration and supported tasks. `dist/Sweep-Setup.exe` bundles
Python/runtime; it does not bundle model weights or user data. Current verified
installed code is `3ae1258be`; later source-only changes do not relabel that binary.
The installer is unsigned. Check `dist/SHA256SUMS.txt` and bundled `build-info.json`
against the intended release before deployment. Checksums establish consistency,
not publisher authenticity.

Close Sweep through its tray menu before upgrade. The normal installer preserves
personal data under `%LOCALAPPDATA%/Sweep`; the app is under
`%LOCALAPPDATA%/Programs/Sweep`. Back up personal state to an access-controlled
location while the app is stopped before any future schema migration. Preserve
model settings/credentials with their OS-user protection; copying encrypted
secrets to another account is not a supported migration. Validate restoration in
an isolated profile before relying on backup. Automatic rollback and enterprise
backup/restore tooling are not implemented.

## Source-only document foundation

Use the existing development environment established by `python setup_sweep.py`.
No new package install or model download is required for this increment. Example
with a file the caller has explicitly selected and granted:

```python
from sweep.documents import inspect_source

selected_path = r"C:\path\to\selected-invoice.pdf"
snapshot = inspect_source(selected_path, [selected_path])
# A caller first checks task/workspace authority and handles errors/partial text.
print(snapshot.text_status, snapshot.review_required, snapshot.empty_pages)
# Reference offsets must come from actual saved text, never fabricated page boxes.
```

This library call does not display an approval prompt; the calling task must
enforce approval and file grants. It does not create a shipment, extract business
fields, reconcile documents, persist a workspace, call an LLM or transmit data.
No personal data migration is performed. Existing desktop results remain unchanged.

Run the focused tests documented in [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)
and `python scripts/benchmark_documents.py --runs 3` for the reproducible local
smoke baseline. Benchmarks are not an installer or a new user-facing chat command.

## Next deployment gates

When routing the new contract through the desktop, test approvals, workspace
membership, worker serialization, old history compatibility, cancellation and
source navigation. Then run `python scripts/build_desktop.py`, verify the native
payload, exercise real workflows outside the repository and upgrade with personal
data preservation checks. Retain honest build revision/dirty-tree metadata.

For any future migration: version schemas, back up/verify counts and hashes,
perform transactional or staged migration, test interruption and restore before
normal use, and never silently rewrite originals. For optional hosting, follow
[SHARED_COMPUTE.md](SHARED_COMPUTE.md); current text hosting lacks native pairing,
multi-machine validation and enterprise document isolation. Do not expose
document processing remotely until the threat-model gates pass.
