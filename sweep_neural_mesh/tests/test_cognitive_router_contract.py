import json,tempfile,unittest
from pathlib import Path
from sweep_neural_mesh.training.train_cognitive_router import contamination,load_rows
class CognitiveRouterContractTests(unittest.TestCase):
 def test_split_validation(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d,"train.jsonl");p.write_text(json.dumps({"split":"train","input_text":"If A then B","domain":"logic","task":"logical_reasoning","evidence":[]})+"\n",encoding="utf-8");rows=load_rows(p,"train");self.assertEqual(len(rows),1);self.assertEqual(contamination(rows,rows)["exact"],1)
 def test_wrong_split_fails(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d,"x.jsonl");p.write_text(json.dumps({"split":"test","input_text":"x","domain":"logic","task":"logical_reasoning"}),encoding="utf-8")
   with self.assertRaises(ValueError):load_rows(p,"train")
if __name__=="__main__":unittest.main()
