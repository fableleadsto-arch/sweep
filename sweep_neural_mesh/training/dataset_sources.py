from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sweep_neural_mesh.training.dataset_pipeline import DatasetEntry, DatasetPipeline
from sweep_neural_mesh.training.safety import DataLicense, SafetyManager


@dataclass(frozen=True)
class DatasetManifest:
    name: str
    platform: str
    reference_url: str
    license_name: str
    license_url: str
    task_types: list[str]
    modalities: list[str]
    default_split: str = "train"
    loader: str = "jsonl"
    input_field: str = "input"
    output_field: str = "expected_output"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImportReport:
    dataset_name: str
    imported: int
    skipped: int
    output_path: str
    audit_entry_id: str
    provenance_items: int
    contamination_free: bool


class OpenSourceDatasetRegistry:
    def __init__(self, safety: SafetyManager | None = None) -> None:
        self._safety = safety or SafetyManager()
        self._manifests: dict[str, DatasetManifest] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        defaults = [
            DatasetManifest(
                name="anli",
                platform="huggingface",
                reference_url="https://huggingface.co/datasets/facebook/anli",
                license_name="facebook-anli",
                license_url="https://huggingface.co/datasets/facebook/anli",
                task_types=["contradiction", "inference", "reasoning"],
                modalities=["text"],
                loader="jsonl",
                input_field="premise",
                output_field="hypothesis",
                metadata={
                    "label_field": "label",
                    "label_map": {"0": "entailment", "1": "neutral", "2": "contradiction"},
                },
            ),
            DatasetManifest(
                name="fever",
                platform="huggingface",
                reference_url="https://huggingface.co/datasets/fever/fever",
                license_name="fever-cc-by-sa-3.0",
                license_url="https://huggingface.co/datasets/fever/fever/blob/main/fever.py",
                task_types=["evidence", "verification", "reasoning"],
                modalities=["text"],
                loader="jsonl",
                input_field="claim",
                output_field="label",
                metadata={"evidence_field": "evidence", "source_field": "id"},
            ),
            DatasetManifest(
                name="logicnli",
                platform="huggingface",
                reference_url="https://huggingface.co/datasets/tasksource/LogicNLI",
                license_name="logicnli-open",
                license_url="https://huggingface.co/datasets/tasksource/LogicNLI/blob/main/README.md",
                task_types=["logic", "reasoning", "first_order_logic"],
                modalities=["text"],
                loader="jsonl",
                input_field="premise",
                output_field="hypothesis",
                metadata={"label_field": "label"},
            ),
            DatasetManifest(
                name="temporal_reasoning_dataset",
                platform="github",
                reference_url="https://github.com/amazon-science/temporal-reasoning-dataset",
                license_name="temporal-reasoning-open",
                license_url="https://github.com/amazon-science/temporal-reasoning-dataset",
                task_types=["temporal", "reasoning"],
                modalities=["text"],
                loader="jsonl",
                input_field="question",
                output_field="answer",
            ),
        ]
        for manifest in defaults:
            self.register_manifest(manifest, allows_commercial=False)

    def register_manifest(
        self,
        manifest: DatasetManifest,
        *,
        allows_commercial: bool = False,
        allows_redistribution: bool = False,
        attribution_required: bool = True,
    ) -> None:
        self._manifests[manifest.name] = manifest
        self._safety.register_license(
            DataLicense(
                dataset_name=manifest.name,
                license_type="open",
                license_url=manifest.license_url,
                allows_training=True,
                allows_commercial=allows_commercial,
                allows_redistribution=allows_redistribution,
                attribution_required=attribution_required,
            )
        )

    def manifests(self) -> list[DatasetManifest]:
        return sorted(self._manifests.values(), key=lambda m: m.name)

    def get(self, name: str) -> DatasetManifest:
        return self._manifests[name]

    def recommended_for(self, task_type: str) -> list[DatasetManifest]:
        matches = [m for m in self._manifests.values() if task_type in m.task_types]
        return sorted(matches, key=lambda m: (m.platform, m.name))

    def import_file(
        self,
        dataset_name: str,
        source_path: str | Path,
        output_dir: str | Path,
        *,
        split: str | None = None,
        source_override: str | None = None,
    ) -> ImportReport:
        manifest = self.get(dataset_name)
        if not self._safety.is_dataset_safe(dataset_name, "training"):
            raise ValueError(f"Dataset '{dataset_name}' is not approved for training")

        pipeline = DatasetPipeline(storage_dir=output_dir)
        path = Path(source_path)
        raw_items = self._load_items(path, manifest.loader)
        imported = 0
        skipped = 0
        license_info = self._safety.check_license(dataset_name)

        for idx, item in enumerate(raw_items):
            entry = self._item_to_entry(
                dataset_name,
                manifest,
                item,
                idx,
                split or manifest.default_split,
                source_override,
            )
            privacy = self._safety.check_privacy(entry.input_text + "\n" + entry.expected_output)
            if privacy["contains_pii"]:
                skipped += 1
                self._safety._log_audit(
                    action="dataset_import",
                    actor="open_source_registry",
                    target=dataset_name,
                    result="denied",
                    reason="PII detected in candidate example",
                    metadata={"entry_id": entry.entry_id, "pii_types": privacy["pii_types"]},
                )
                continue
            if pipeline.add_entry(entry):
                imported += 1
                self._safety.track_provenance(
                    item_id=entry.entry_id,
                    source=source_override or manifest.reference_url,
                    license=manifest.license_name,
                    allowed_uses=["training", "evaluation"],
                    restrictions=["attribution_required"] if license_info and license_info.attribution_required else [],
                )
            else:
                skipped += 1

        contamination = pipeline.check_contamination()
        export_path = pipeline.export_jsonl(split or manifest.default_split)
        audit = self._safety.log_data_access(dataset_name, "open_source_registry", "training_import")
        return ImportReport(
            dataset_name=dataset_name,
            imported=imported,
            skipped=skipped,
            output_path=export_path,
            audit_entry_id=audit.entry_id,
            provenance_items=imported,
            contamination_free=bool(contamination["contamination_free"]),
        )

    def export_manifest_catalog(self, path: str | Path) -> str:
        rows = []
        for manifest in self.manifests():
            lic = self._safety.check_license(manifest.name)
            rows.append(
                {
                    "name": manifest.name,
                    "platform": manifest.platform,
                    "reference_url": manifest.reference_url,
                    "license_name": manifest.license_name,
                    "license_url": manifest.license_url,
                    "allows_training": bool(lic and lic.allows_training),
                    "allows_commercial": bool(lic and lic.allows_commercial),
                    "attribution_required": bool(lic and lic.attribution_required),
                    "task_types": manifest.task_types,
                    "modalities": manifest.modalities,
                }
            )
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"generated_at": time.time(), "datasets": rows}, indent=2), encoding="utf-8")
        return str(out)

    def _load_items(self, path: Path, loader: str) -> list[dict[str, Any]]:
        if loader == "jsonl":
            items: list[dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
            return items
        if loader == "json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [dict(x) for x in data]
            if isinstance(data, dict):
                payload = data.get("data") or data.get("examples") or []
                return [dict(x) for x in payload]
        raise ValueError(f"Unsupported loader '{loader}'")

    def _item_to_entry(
        self,
        dataset_name: str,
        manifest: DatasetManifest,
        item: dict[str, Any],
        index: int,
        split: str,
        source_override: str | None,
    ) -> DatasetEntry:
        input_text = self._compose_input(manifest, item)
        expected_output = self._extract_output(manifest, item)
        quality = float(item.get("quality", 0.8))
        difficulty = int(item.get("difficulty", 2))
        metadata = dict(manifest.metadata)
        metadata.update(item.get("metadata", {}))
        label_field = str(manifest.metadata.get("label_field", "label"))
        if label_field in item:
            metadata["label"] = item.get(label_field)
        return DatasetEntry(
            entry_id=str(item.get("id") or item.get("uid") or f"{dataset_name.upper()}-{index:06d}"),
            task_type=manifest.task_types[0],
            difficulty=difficulty,
            modality=manifest.modalities[0],
            source=source_override or manifest.reference_url,
            license=manifest.license_name,
            quality=quality,
            input_text=input_text,
            expected_output=expected_output,
            evaluation_criteria=str(item.get("evaluation_criteria", "exact_match")),
            split=split,
            metadata=metadata,
        )

    def _compose_input(self, manifest: DatasetManifest, item: dict[str, Any]) -> str:
        if manifest.name in {"anli", "logicnli"}:
            premise = str(item.get("premise", "")).strip()
            hypothesis = str(item.get("hypothesis", "")).strip()
            return f"Premise: {premise}\nHypothesis: {hypothesis}".strip()
        if manifest.name == "fever":
            claim = str(item.get("claim", "")).strip()
            evidence = item.get("evidence") or []
            evidence_text = " | ".join(str(x) for x in evidence[:3])
            return f"Claim: {claim}\nEvidence: {evidence_text}".strip()
        return str(item.get(manifest.input_field, item.get("input", item.get("input_text", "")))).strip()

    def _extract_output(self, manifest: DatasetManifest, item: dict[str, Any]) -> str:
        if manifest.name in {"anli", "logicnli"}:
            label = item.get(str(manifest.metadata.get("label_field", "label")), item.get("label"))
            label_map = manifest.metadata.get("label_map") or {}
            return str(label_map.get(str(label), label)).strip()
        return str(item.get(manifest.output_field, item.get("expected_output", item.get("answer", "")))).strip()
