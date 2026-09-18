import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from sweep_neural_mesh.workloads import data_analyze,document_extract,numeric_describe,numeric_simulate
class WorkloadTests(unittest.TestCase):
 def test_document_hash_and_query(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d,"note.txt");p.write_text("Alpha\nBeta target\n",encoding="utf-8");out=document_extract(str(p),"target");self.assertEqual(out["status"],"completed");self.assertIn("Beta target",out["content"]);self.assertEqual(len(out["sha256"]),64)
 def test_csv_analysis(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d,"data.csv");p.write_text("team,value\na,1\na,3\nb,2\n",encoding="utf-8");out=data_analyze(str(p),"team");self.assertEqual(out["result"]["rows"],3);self.assertEqual(out["result"]["group_counts"]["a"],2);self.assertEqual(out["result"]["numeric_summary"]["value"]["mean"],2.0)
 def test_numeric_and_bounded_simulation(self):
  self.assertEqual(numeric_describe([1,2,3])["result"]["median"],2);self.assertEqual(len(numeric_simulate("random-walk",7)["series"]),7)
 def test_research_redirect_is_not_followed(self):
  class R:
   status_code=302;content=b"";text=""
  with patch("httpx.get",return_value=R()):
   from sweep_neural_mesh.workloads import research_fetch
   out=research_fetch("https://example.com");self.assertEqual(out["status"],"blocked")
if __name__=="__main__":unittest.main()
