import tempfile,unittest
from pathlib import Path
from sweep_neural_mesh.cognition import CognitiveLoop,Observation,ReasoningStore
class CognitiveLoopTests(unittest.TestCase):
 def build(self,path,verification):return CognitiveLoop(lambda q:{"domain":"logic","task":"logical_reasoning"},lambda ctx:{"text":"A follows from B"},lambda *args:verification,ReasoningStore(path))
 def test_unsupported_generation_stays_unresolved(self):
  with tempfile.TemporaryDirectory() as d:
   c=self.build(Path(d,"reasoning.jsonl"),{}).run("question");self.assertEqual(c.status,"unresolved");self.assertEqual(c.confidence,0.0)
 def test_independent_evidence_supports_and_persists_explanation(self):
  with tempfile.TemporaryDirectory() as d:
   loop=self.build(Path(d,"reasoning.jsonl"),{"verified":True,"confidence":.82,"evidence_ids":["ev1","ev2"],"independent_sources":2});c=loop.run("question",[Observation("source-a",{"fact":1},confidence=.9)]);self.assertEqual(c.status,"supported");self.assertEqual(c.confidence,.82);explanation=loop.explain_conclusion(c.id);self.assertEqual(len(explanation["events"]),7);self.assertEqual(explanation["conclusion"]["id"],c.id)
 def test_single_source_confidence_is_capped(self):
  with tempfile.TemporaryDirectory() as d:
   c=self.build(Path(d,"r.jsonl"),{"verified":True,"confidence":.95,"evidence_ids":["ev1"],"independent_sources":1}).run("question");self.assertEqual(c.confidence,.65)
 def test_event_budget_is_bounded(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):CognitiveLoop(lambda x:{},lambda x:{},lambda *x:{},ReasoningStore(Path(d,"r")),9)
if __name__=="__main__":unittest.main()
