import json,tempfile,unittest
from pathlib import Path
from sweep_neural_mesh.training.import_clinc150 import normalize
from sweep_neural_mesh.training.source_policy import SourcePolicy
class ExternalSourceTests(unittest.TestCase):
 def test_bigbench_is_blocked_from_training(self):
  with self.assertRaises(PermissionError):SourcePolicy().require_training_allowed("bigbench_logical_deduction_three_objects")
 def test_clinc_fixture_preserves_splits(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);source=root/"data.json";source.write_text(json.dumps({"train":[["find my card","card_arrival"]],"val":[["where is it","card_arrival"]],"test":[["track card","card_arrival"]],"oos_train":[["sing a song","oos"]],"oos_val":[],"oos_test":[]}));report=normalize(source,root/"out",allow_unpinned=True);self.assertEqual(report["counts"]["train"],1);self.assertTrue((root/"out"/"clinc150_test.jsonl").exists())
