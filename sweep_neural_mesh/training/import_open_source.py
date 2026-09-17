from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from sweep_neural_mesh.training.dataset_sources import OpenSourceDatasetRegistry
from sweep_neural_mesh.training.safety import SafetyManager


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import approved open-source dataset files into SWEEP's governed DatasetPipeline format."
    )
    parser.add_argument("--dataset", required=True, help="Registered dataset manifest name, e.g. anli, fever, logicnli")
    parser.add_argument("--input", required=True, help="Local JSONL/JSON file to import. Downloading is intentionally external/explicit.")
    parser.add_argument("--output-dir", default="sweep_neural_mesh/training/datasets/open_source", help="Output directory for normalized JSONL")
    parser.add_argument("--audit-dir", default="sweep_neural_mesh/training/audit", help="Audit directory")
    parser.add_argument("--split", default=None, help="Optional fixed split label. Defaults to manifest split.")
    parser.add_argument("--source-url", default=None, help="Optional exact source URL for this imported file")
    parser.add_argument("--catalog", action="store_true", help="Export the approved source catalog and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    safety = SafetyManager(audit_dir=args.audit_dir)
    registry = OpenSourceDatasetRegistry(safety)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.catalog:
        catalog_path = registry.export_manifest_catalog(out_dir / "open_source_dataset_catalog.generated.json")
        print(json.dumps({"ok": True, "catalog": catalog_path}, indent=2))
        return 0

    report = registry.import_file(
        args.dataset,
        args.input,
        out_dir,
        split=args.split,
        source_override=args.source_url,
    )
    print(json.dumps({"ok": True, "report": asdict(report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
