# TradeOps evaluation protocol

2026-10-28. Status: **Partially implemented**: contract/fixture tests first;
representative customer accuracy has not been measured.

## Reproducibility and data rights

Record code revision, environment/dependency versions, corpus/fixture hashes,
seed where generation uses randomness, schema/rule/model versions, hardware and
full command. Use source-controlled synthetic text/PDF/DOCX fixtures with visibly
fictional entities and a SYNTHETIC TEST DOCUMENT label. Fixed fixtures are already
deterministic; a seed is not evidence of representative data. No real proprietary
or personal information is needed for foundation tests.

External data must pass the [resource registry](SOURCE_AND_DATASET_REGISTRY.md)
rights/access review. Keep authorized customer evaluation distinct from public
benchmarks and synthetic regression. Split by vendor/template/document family;
near-duplicate documents cannot cross train/development/test partitions. Freeze
test labels before scoring; track corrections and dataset revisions.

## Layers and gates

1. **Foundation invariants:** real local reader outputs and unchanged input hashes;
   exact page/unit/range resolution; strict serialized types and schema versions;
   reject swapped documents, altered extracts/quotes and invalid ranges. Verify
   blank/partial extraction, bounds and Unicode behavior without model calls.
2. **Extraction:** compare typed normalized critical fields against independently
   reviewed labels. A missing answer for a present field is incorrect. Count
   ambiguous/unsupported outcomes separately and include them in coverage. Score
   absent-field correctness separately to avoid accuracy dominated by nulls.
3. **Reconciliation:** evaluate independently labeled document pairs, matching and
   discrepancy types. TP means correct rule, line/field identity and source pair;
   duplicated findings cannot inflate TP. Measure precision TP/(TP+FP), recall
   TP/(TP+FN), abstentions and high-severity strata. Zero predictions means precision
   is undefined, not 100%. Missing required fields and contradictions are separate.
4. **Provenance/corrections:** every material finding resolves to exact authorized
   source references. Measure citation coverage and correctness, not merely presence
   of a citation string. Corrected values retain originals/actor/revision; stale
   approvals fail. Human corrections do not count as automated extraction successes.
5. **Workflow/security:** native review, export and exact-action approval; interrupted
   worker, denial, malformed documents, cross-workspace access and original-file
   preservation. No unsupported placeholder control counts as completion.
6. **Deployment/performance:** source then frozen/installed workflows outside the
   repository, personal-data preservation, offline operation, cancellation and
   resource measurements. UI tests include actual source navigation and long values.

## Metrics and reporting

Proposed pilot gates: >=95% exact match for a declared critical-field set and
>=98% precision for high-severity discrepancies. Report numerator/denominator,
documents, families, languages, pages, scan fraction and unsupported fraction.
Use Wilson 95% intervals for proportions; never infer population performance from
small synthetic tests. Track recall alongside precision and obtain customer review
of severity definitions. No current test suite establishes these targets.

For timing, distinguish cold process/parser startup from repeated extraction and
from model warm/cold inference. Record each sample, failures and input sizes;
report median/p95 only with a stated sample size (at least 30 for the initial
performance study). A small smoke benchmark reports raw samples, not a reliable
p95. Measure total process-tree peak/idle RAM separately from Python allocations.
Monetary costs follow [COST_AND_UNIT_ECONOMICS.md](COST_AND_UNIT_ECONOMICS.md).

Initial test command:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_documents.py tests/test_desktop_data_tools.py tests/test_desktop_document_tasks.py tests/test_desktop_results.py
```

Actual counts/results belong in the continuation state after execution. Remaining
pilot-data and deployment gaps stay explicit.
