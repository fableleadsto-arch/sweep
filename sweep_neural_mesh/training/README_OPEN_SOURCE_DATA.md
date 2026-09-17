# Open-source data import for SWEEP training

This directory now has a governed import path for public/open-source training data.

## What this does

- Registers approved dataset manifests in `dataset_sources.py`.
- Normalizes local JSONL/JSON files into SWEEP `DatasetPipeline` entries.
- Checks the registered license metadata before import.
- Screens examples for basic PII before adding them to training data.
- Tracks provenance for imported examples.
- Writes audit entries for allowed and denied import actions.

## What this does not do

- It does not silently scrape arbitrary websites.
- It does not download private, leaked, restricted, paywalled, or authentication-protected data.
- It does not train a model by itself.
- It does not prove 90%+ results.

## Usage

Export the governed source catalog:

```bash
python -m sweep_neural_mesh.training.import_open_source \
  --dataset anli \
  --input unused.jsonl \
  --catalog
```

Import a local approved source file:

```bash
python -m sweep_neural_mesh.training.import_open_source \
  --dataset anli \
  --input /path/to/anli_sample.jsonl \
  --output-dir sweep_neural_mesh/training/datasets/open_source \
  --audit-dir sweep_neural_mesh/training/audit
```

The next required step is to download or mount actual approved dataset files, verify their licenses in detail, import them, freeze train/validation/test splits, and then run a held-out evaluation. Until that happens, do not claim that SWEEP has been trained on these external sources or reached any target accuracy.
