PHASE 5 MILESTONE REPORT
Phase: Native acceleration hardening
Milestone: C++ dataframe serialization repair and opt-in extension build
Status: PASS
Implementation:
- Repaired string handling in `dataframe_to_csv`, `df_drop_duplicates`, and `df_group_by`.
- Numeric conversion is now selected only for arithmetic variant alternatives; strings are preserved as strings.
- Added an `Entity::length()` accessor required by the existing NLP span calculations.
- Repaired malformed HTML parser raw-string regex literals and helper ordering.
- Replaced non-portable `std::regex::pattern()` reporting with matched-signal reporting.
- Added the missing `<unordered_set>` include for search-hit deduplication.
- Added a native harness covering strings, doubles, booleans, missing values, duplicate removal, and grouping.
- Added an opt-in GitHub Actions job for translation-unit compilation, runtime harness execution, and full `SWEEP_BUILD_CPP=1` package installation.
Tests: Native diagnostics persisted by CI at commit `e8cfb523a03883a2b21b080fc3705c527cb1ca5b`.
Integration: Python installation remains the default; native acceleration is still explicitly opt-in.
Documentation: This report records the scope and evidence boundary.
Tests passed: Harness compile exit `0`; harness runtime exit `0`; full opt-in native package build exit `0`; GitGuardian passed.
Tests failed: None in the validated Linux CI path.
Demonstration: `g++ -std=c++20 -Icpp/include cpp/src/data_engine.cpp cpp/tests/test_data_engine.cpp` followed by the harness executable; `SWEEP_BUILD_CPP=1 python -m pip install -e .`.
Measured results: Native dataframe harness compiled and ran successfully; the full editable package with the native extension built successfully in dependency-complete CI.
Baseline: `SWEEP_BUILD_CPP=1` failed at string `std::to_string` calls and then exposed independent HTML, NLP, regex, and search-ranker compile defects.
Current: The complete Linux opt-in native package build passes, while the Python fallback remains installable by default.
Target: Add platform-matrix native tests and runtime API smoke tests for the compiled extension.
Known limitations: Only the dependency-complete Linux CI path is verified; no Windows/macOS compiler matrix or performance claim is made.
Known failures: No known failures in this checkpoint.
Dependencies satisfied: C++20 compiler, pybind11 build configuration, existing optional-build switch, native regression harness.
Dependencies remaining: Platform matrix, extension import/runtime smoke test, and performance benchmarking.
Ready for dependent phases: YES for further CLI and adapter work; NO for claims of cross-platform native readiness.
