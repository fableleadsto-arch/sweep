PHASE 2 MILESTONE REPORT
Phase: External dataset training
Milestone: Checkpoint 9 — pinned CLINC150 GitHub Actions training run
Status: PARTIAL
Implementation:
- Added a branch-scoped workflow that downloads only the pinned CLINC150 revision and verifies its Git blob.
- Trains word/bigram TF-IDF plus LinearSVC on upstream train and oos_train only.
- Evaluates unchanged validation/test and OOS splits.
- Records exact-text overlap contamination audit, majority baseline, accuracy, macro-F1, in-scope accuracy, OOS recall, seed, environment, and duration.
- Uploads model and metrics as 30-day CI artifacts; weights are not committed.
Tests:
- Trainer compiled locally.
- Full execution delegated to GitHub runner because local DNS and scikit-learn installation are unavailable.
Integration:
- Uses the source lock and CLINC normalizer from checkpoint 8.
Documentation: This report.
Tests passed: Local compile.
Tests failed: Local fixture execution blocked by missing scikit-learn; CI installs it explicitly.
Demonstration: Inspect the CLINC150 pinned training workflow for this commit.
Measured results: Pending CI completion; no score claimed yet.
Baseline: Pending computed train-majority fraction.
Current: Pending immutable validation/test evaluation.
Target: Measured external baseline without contamination or fabricated claims.
Known limitations: First model is a bounded CPU baseline, not the full neural mesh.
Known failures: Direct sandbox download and pip installation failed due DNS.
Dependencies satisfied: Exact source pin, license, split policy, deterministic seed, CI runtime definition.
Dependencies remaining: Successful CI run and artifact inspection.
Ready for dependent phases: NO until CI result is verified.
