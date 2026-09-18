import json,tempfile,unittest
from pathlib import Path
from sweep_neural_mesh.report_workloads import report_markdown
class ReportWorkloadTests(unittest.TestCase):
 def test_report_preserves_input_provenance(self):
  with tempfile.TemporaryDirectory() as d:
   source=Path(d,"result.json");target=Path(d,"report.md");source.write_text(json.dumps({"status":"completed","summary":"sample","result":{"value":3}}),encoding="utf-8")
   out=report_markdown(str(source),str(target));self.assertEqual(out["status"],"completed");self.assertFalse(out["claims_verified"]);self.assertIn("sample",target.read_text(encoding="utf-8"));self.assertEqual(len(out["input"]["sha256"]),64)
 def test_report_rejects_same_path(self):
  with tempfile.TemporaryDirectory() as d:
   source=Path(d,"result.json");source.write_text("{}",encoding="utf-8");out=report_markdown(str(source),str(source));self.assertEqual(out["status"],"error")
if __name__=="__main__":unittest.main()
