from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sweep_neural_mesh.training.dataset_sources import DatasetManifest, OpenSourceDatasetRegistry
from sweep_neural_mesh.training.safety import DataLicense, SafetyManager


class DatasetSourcesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.safety = SafetyManager(audit_dir=self.root / "audit")
        self.registry = OpenSourceDatasetRegistry(self.safety)

    def test_default_manifests_include_multiple_platforms(self) -> None:
        names = [m.name for m in self.registry.manifests()]
        self.assertIn("anli", names)
        self.assertIn("fever", names)
        self.assertIn("temporal_reasoning_dataset", names)
        self.assertTrue(any(m.platform == "huggingface" for m in self.registry.manifests()))
        self.assertTrue(any(m.platform == "github" for m in self.registry.manifests()))

    def test_recommended_for_logic_returns_logicnli(self) -> None:
        matches = [m.name for m in self.registry.recommended_for("logic")]
        self.assertIn("logicnli", matches)

    def test_import_anli_jsonl_normalizes_and_tracks_provenance(self) -> None:
        sample = self.root / "anli.jsonl"
        sample.write_text(
            json.dumps({"uid": "anli-1", "premise": "All birds can fly.", "hypothesis": "Penguins can fly.", "label": 2}) + "\n",
            encoding="utf-8",
        )
        out_dir = self.root / "dataset_out"
        report = self.registry.import_file("anli", sample, out_dir)
        self.assertEqual(report.imported, 1)
        self.assertTrue(report.contamination_free)
        exported = Path(report.output_path).read_text(encoding="utf-8")
        self.assertIn("contradiction", exported)
        self.assertEqual(len(self.safety._provenance), 1)

    def test_import_skips_pii(self) -> None:
        sample = self.root / "fever.jsonl"
        sample.write_text(
            json.dumps({"id": "f1", "claim": "Contact me at test@example.com", "label": "SUPPORTS", "evidence": ["public note"]}) + "\n",
            encoding="utf-8",
        )
        out_dir = self.root / "dataset_out"
        report = self.registry.import_file("fever", sample, out_dir)
        self.assertEqual(report.imported, 0)
        self.assertEqual(report.skipped, 1)

    def test_restricted_dataset_is_blocked(self) -> None:
        self.registry.register_manifest(
            DatasetManifest(
                name="blocked_ds",
                platform="github",
                reference_url="https://example.com/blocked",
                license_name="blocked",
                license_url="https://example.com/license",
                task_types=["logic"],
                modalities=["text"],
            )
        )
        self.safety.register_license(
            DataLicense(dataset_name="blocked_ds", license_type="restricted", allows_training=False)
        )
        sample = self.root / "blocked.jsonl"
        sample.write_text(json.dumps({"input": "x", "expected_output": "y"}) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.registry.import_file("blocked_ds", sample, self.root / "out")

    def test_manifest_catalog_exports_license_flags(self) -> None:
        path = self.root / "catalog.json"
        out = self.registry.export_manifest_catalog(path)
        data = json.loads(Path(out).read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(data["datasets"]), 4)
        self.assertTrue(all("allows_training" in row for row in data["datasets"]))


if __name__ == "__main__":
    unittest.main()
