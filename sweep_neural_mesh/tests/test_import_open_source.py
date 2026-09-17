from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from sweep_neural_mesh.training.import_open_source import main


class ImportOpenSourceCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_cli_imports_approved_dataset_file(self) -> None:
        source = self.root / "anli.jsonl"
        source.write_text(
            json.dumps({"uid": "x1", "premise": "A is true.", "hypothesis": "A is true.", "label": 0}) + "\n",
            encoding="utf-8",
        )
        out = self.root / "out"
        audit = self.root / "audit"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["--dataset", "anli", "--input", str(source), "--output-dir", str(out), "--audit-dir", str(audit)])
        self.assertEqual(code, 0)
        payload = json.loads(buf.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["report"]["imported"], 1)
        self.assertTrue(Path(payload["report"]["output_path"]).exists())

    def test_cli_exports_catalog(self) -> None:
        out = self.root / "out"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main([
                "--dataset", "anli",
                "--input", str(self.root / "unused.jsonl"),
                "--output-dir", str(out),
                "--audit-dir", str(self.root / "audit"),
                "--catalog",
            ])
        self.assertEqual(code, 0)
        payload = json.loads(buf.getvalue())
        self.assertTrue(payload["ok"])
        self.assertTrue(Path(payload["catalog"]).exists())


if __name__ == "__main__":
    unittest.main()
