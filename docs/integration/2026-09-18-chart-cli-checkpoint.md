PHASE 8 MILESTONE REPORT
Phase: Multipurpose data workload expansion
Milestone: Governed chart generation through the existing data tool
Status: PASS
Implementation:
- Added `sweep data plot FILE --kind {line,bar,scatter,hist} --output PNG`.
- Reused `companion.tools.data.run_plot`; no parallel plotting engine was created.
- Added input size bounds, source SHA-256, required output paths, output SHA-256, and explicit optional-dependency failure states.
- Bumped the unified CLI to `sweep-cli/7`.
Tests: CI diagnostics persisted at `docs/metrics/cli_v7_ci_exit_code.txt`.
Tests passed: Editable install; 20/20 tests; clean wheel build/install; isolated `sweep --version`; CLI safety smoke commands; CI exit code `0`; GitGuardian passed.
Measured results: The plotting adapter produces a real PNG when the existing Matplotlib-backed tool is available; otherwise it returns `unavailable` without fabricating an artifact.
Demonstration: `sweep data plot data.csv --kind scatter --x time --y value --output chart.png`.
Known limitations: No chart-quality benchmark is claimed; plotting remains dependent on optional Matplotlib availability and local data quality.
Dependencies remaining: Report generation, richer transformation adapters, representative chart evaluation, and broader platform coverage.
Ready for dependent phases: YES for further data/report workload development; NO for quality or production-readiness claims.
