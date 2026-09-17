from __future__ import annotations
import json,tempfile,unittest
from pathlib import Path
from sweep_neural_mesh.intelligence_agent import CompatibleFileMemory,IntelligenceRouter,extract,plan
class IntelligenceAgentTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.router=IntelligenceRouter(Path(self.tmp.name)/"artifacts"/"intelligence_router.json")
 def tearDown(self):self.tmp.cleanup()
 def test_routes_intelligence_tasks(self):
  cases={"conduct due diligence on Northwind":"investigation","reconstruct the sequence of events":"timeline","detect inconsistent statements":"contradiction_analysis","parse people organizations and places":"extraction","launch chrome":"device_action"}
  for text,expected in cases.items():self.assertEqual(self.router.classify(text)["intent"],expected)
 def test_extraction_is_observation(self):
  result=extract("Alice Smith emailed a@example.com on 2026-09-17. See https://example.com");self.assertIn("a@example.com",result["emails"]);self.assertIn("2026-09-17",result["dates"])
 def test_device_action_never_fakes_success(self):self.assertEqual(plan("launch chrome",{"intent":"device_action","confidence":1.0})["result"]["status"],"unavailable")
 def test_memory_uses_existing_schema(self):
  with tempfile.TemporaryDirectory() as td:
   path=Path(td)/"memory.json";memory=CompatibleFileMemory(path);memory.remember("Project Atlas uses public sources");self.assertIn("Project Atlas",memory.recall("Atlas")[0]["content"]);self.assertEqual(json.loads(path.read_text())["version"],1)
if __name__=="__main__":unittest.main()
