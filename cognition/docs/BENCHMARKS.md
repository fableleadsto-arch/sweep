# Sweep Cognition — Benchmarking

`cognition/bench.py` runs a deterministic, self-contained capability battery
and reports **all twelve required measurements**. No LLM is involved; inputs
are fixed known-answer tasks, so the benchmark is reproducible run-to-run.

## The measurements

| Measurement                 | What it measures                                   |
|-----------------------------|----------------------------------------------------|
| `logical_accuracy`          | % of transitive/symmetric logic known-answer queries correct |
| `evidence_attribution`      | claim -> evidence -> source provenance linkup     |
| `contradiction_detection`   | rule-based contradiction pairs correctly flagged/cleared |
| `memory_retrieval`          | long-term memory recall of known facts            |
| `hypothesis_ranking`        | correct hypothesis rises to rank 1                |
| `belief_revision`           | leading belief switches with history retained     |
| `uncertainty_calibration`   | each epistemic state produced exactly as expected |
| `hallucination_error_rate`  | ambiguous inputs wrongly declared VERIFIED (target 0%) |
| `router_accuracy`           | task classification matches expected route        |
| `latency_ms`                | average end-to-end loop time per task             |
| `cpu_usage`                 | measured CPU percent during benchmark (psutil)    |
| `ram_usage_mb`              | measured RSS during benchmark                     |

## Running

```bash
python -m cognition.bench                      # prints JSON
python -c "from cognition.bench import run_benchmark, write_report; \
write_report(run_benchmark(), 'benchmark/cognition_benchmark.json')"
```

The report is also the authorship evidence for `IMPLEMENTATION_STATUS.md`.

## Latest recorded results

Written to `benchmark/cognition_benchmark.json` during Phase 19 (reproduced in
`IMPLEMENTATION_STATUS.md`). Capability metrics run at 100% ceiling and
hallucination rate at 0.0%, confirming the deterministic guarantees of the
cognition package.