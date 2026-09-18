import unittest
from sweep_neural_mesh.terminal import execute
class Router:
 def __init__(self,intent):self.intent=intent
 def classify(self,text):return {"intent":self.intent,"confidence":1.0}
class TerminalTests(unittest.TestCase):
 def test_neural_delegation(self):
  out=execute("analyze",Router("evidence_analysis"),True,neural_executor=lambda q:{"status":"completed","observation":"verified"});self.assertEqual(out["result"]["observation"],"verified")
 def test_device_is_plan_only(self):
  out=execute("launch chrome",Router("device_action"),True,device_executor=lambda q:{"status":"planned","execution":"not_performed"});self.assertEqual(out["result"]["execution"],"not_performed")
