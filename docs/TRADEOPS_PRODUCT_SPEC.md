# TradeOps initial product contract

2026-10-28. Product status: **Planned**; source-linked ingestion foundation is
implemented and tested as a source-only API. This is not a shipped MVP.
Governing scope: [master directive](directives/SWEEP_ENTERPRISE_MASTER.md).

## First complete workflow

A user attaches one commercial invoice and one packing list for the same shipment
in the native chat, reviews their document roles, and requests comparison. Sweep
extracts available fields locally, identifies discrepancies with both sources,
accepts explicit corrections without changing original evidence, and saves a
reviewable report. A draft request for corrected documents may follow. Sending
that draft or updating an existing spreadsheet requires separate exact-action
approval. No customs submission, classification decision or portal automation is
implicitly authorized by document attachment.

Start with English text-bearing PDFs and clearly labeled text fixtures. DOCX
extraction is useful but has no reliable physical pagination. Scans, arbitrary
tables, handwriting, other languages and ambiguous layouts require an explicit
unsupported/manual-review result until measured adapters support them. Do not
quietly treat a partial preview as a complete shipment record. Excel import/export
is a subsequent tested adapter, not satisfied by renaming a CSV.

## Shared source contract

The first additive API is `sweep.documents`: a bounded, immutable extraction
snapshot over the current `desktop.data_tools.inspect_document` reader. Preserve
schema version, original SHA-256/byte count, format, parser identity/version,
extracted units and truncation. PDF units retain actual one-based page numbers;
DOCX/text units have no invented page number. No bounding box is fabricated.
`empty_pages` identifies PDF pages without selectable text, including mixed PDFs;
these may be blank or need OCR, so `review_required` remains true even if other
pages contain text. This flag is an extraction limitation, not a business verdict.
Text offsets are zero-based Python Unicode character indices, end-exclusive, in
the saved extraction unit, not byte offsets or original-file coordinates.

A reference binds original-file hash, extraction hash, unit index, character
range and exact quote. Resolving it verifies all components against the snapshot.
The extraction hash also changes when parser version, page text, order or
completeness changes. Hashes detect mismatches; they are not signatures, access
tokens, proof of authenticity or permission to reopen a file. The caller must
enforce workspace membership before resolving a reference. Existing task/chat
results stay readable; this additive contract does not migrate user records.

## Field and correction contract (next increments)

Each field will retain a stable ID, document role, field name, raw value, normalized
value/type, one or more verified references, extraction method/version, uncertainty
state (`observed`, `ambiguous`, `missing`, `unsupported`), validation findings and
revision history. Missing values are null, not inferred defaults. An observed
value is not a calibrated probability. Multiple conflicting candidates remain
visible; never choose a convenient one silently.

Initial critical fields: document number, shipment/order reference, buyer/seller,
line identifier, description, quantity/unit, currency, unit price and line/invoice
totals where supplied. Expand to addresses/countries/dates, freight/insurance,
origin/destination, supplied HS codes/Incoterms and container/tracking references.
Packing lists need not contain prices; required fields are role-specific.

Corrections retain old/new values, actor, UTC timestamp, reason, revision and
original references. A correction is a user assertion, not extracted evidence.
Reconciliation records exact field revisions. Concurrent/stale changes must be
rejected or explicitly reviewed; changing a value invalidates earlier approval
of an exported or external action. Originals remain unchanged.

## Deterministic reconciliation rules

| Rule | Behavior / uncertainty |
| --- | --- |
| Shipment association | User-confirmed pair; different supplied references are flagged, missing references require review. |
| Line matching | Unique supplied line/SKU identity first; duplicates or several plausible matches require review. Never auto-match by list order. |
| Quantity | Compare exact Decimal values only for matched lines and compatible explicit units. Unknown units or ambiguous separators require review. |
| Price/totals | Decimal multiplication/sum with documented currency rounding; account for explicitly supplied discounts/taxes/freight. Missing components cannot prove a total wrong. |
| Currency | Compare explicit currency codes; ambiguous currency symbols require review. No implicit exchange-rate conversion. |
| Names/addresses/descriptions | Preserve originals; whitespace normalization allowed. Fuzzy similarity yields a review suggestion, not a confirmed match/mismatch. |
| Dates | Preserve source and locale assumptions; ambiguous dates require review. Apply documented workflow constraints, not invented legal deadlines. |
| Duplicates/versions | Distinguish identical file bytes, repeated line IDs and revised documents. Same bytes in another workspace do not grant access. |
| Missing fields | Report role-required missing/unsupported fields separately from contradictions. Truncated extraction cannot establish absence in the original. |

A discrepancy stores rule/version, severity, status, linked field revisions,
expected/observed values and explanation. High severity is limited initially to
unambiguous matched-line quantity/currency contradictions and fully calculable
totals. Report false positives and missed discrepancies; do not boost precision by
concealing abstentions.

## Native review and acceptance examples

The dock stays the entry point. An expanded native review panel shows documents
side by side, extracted fields and a discrepancy list. Selecting a finding opens
the exact saved source/page and highlights its text span; a missing page image or
box is stated explicitly. Review/correction/report actions work with keyboard
navigation, long values, narrow windows, cancellations and interrupted tasks.
Plain text from documents cannot become HTML, shell commands or authorization.

- Synthetic invoice SKU-A quantity 12 versus packing quantity 10: one quantity
  finding citing both actual spans; no model arithmetic required.
- Same documents with quantity 12: no quantity finding, assuming same explicit unit.
- Invoice uses boxes and packing uses pieces with no supplied conversion: manual
  review; never report the arithmetic difference as a confirmed quantity error.
- Scanned/blank or truncated page: extraction limitation; no fabricated shipment.
- Correction 10 -> 12: retain extracted 10 and actor correction; rerun against the
  corrected revision, preserving the earlier result and audit history.
- Foreign-workspace source, replaced snapshot or wrong quote: deny resolution.
- Denied send/update approval: no external action and no source modification.

Report JSON is versioned and source-linked; CSV/Excel are reviewable projections
with spreadsheet formula-injection defenses. Saved drafts are clearly drafts.
The full workflow is accepted only after source, native UI, packaged and installed
tests and the [evaluation protocol](EVALUATION_PROTOCOL.md); library tests alone
do not mark the MVP complete.
