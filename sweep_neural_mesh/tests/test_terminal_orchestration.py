import sys,types,unittest
from sweep_neural_mesh.terminal_orchestration import query_neural
class TerminalOrchestrationTests(unittest.TestCase):
 def _install(self,result):
  module=types.ModuleType("sweep_neural_mesh.sweep_api")
  class API:
   def query(self,q):return result
  module.SweepAPI=API;sys.modules["sweep_neural_mesh.sweep_api"]=module
 def test_success_is_actual_observation(self):
  self._install(types.SimpleNamespace(method="logic_engine",confidence=.7,answer="supported",reasoning="rule",components=["logic"]));r=query_neural("q");self.assertEqual(r["status"],"completed");self.assertEqual(r["observation"],"supported")
 def test_unavailable_is_not_success(self):
  self._install(types.SimpleNamespace(method="unavailable",confidence=0,answer="UNKNOWN",reasoning="",components=[]));self.assertEqual(query_neural("q")["status"],"unavailable")
