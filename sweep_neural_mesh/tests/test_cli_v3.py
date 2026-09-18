import io,json,tempfile,unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from sweep_neural_mesh.terminal import execute,main
class Router:
 def __init__(self,intent):self.intent=intent
 def classify(self,text):return {"intent":self.intent,"confidence":1.0}
class CliV3Tests(unittest.TestCase):
 def capture(self,args):
  stream=io.StringIO()
  with redirect_stdout(stream):code=main(args)
  return code,json.loads(stream.getvalue())
 def test_legacy_execute_neural_delegation(self):
  out=execute("analyze",Router("evidence_analysis"),True,neural_executor=lambda q:{"status":"completed","observation":"verified"});self.assertEqual(out["result"]["observation"],"verified")
 def test_models_command_probes_without_loading_weights(self):
  with patch("sweep_neural_mesh.terminal.model_status",return_value={"general":{"status":"missing"}}):code,out=self.capture(["models"])
  self.assertEqual(code,0);self.assertEqual(out["general"]["status"],"missing")
 def test_explain_missing_is_explicit(self):
  with tempfile.TemporaryDirectory() as d:code,out=self.capture(["explain","--reasoning-file",str(Path(d,"none.jsonl")),"missing"])
  self.assertEqual(code,2);self.assertEqual(out["status"],"not_found")
 def test_device_execution_bypass_is_blocked(self):
  code,out=self.capture(["device","run"]);self.assertEqual(code,3);self.assertEqual(out["status"],"blocked")
 def test_evaluate_missing_is_explicit(self):
  with tempfile.TemporaryDirectory() as d:code,out=self.capture(["evaluate","cognitive-router","--metrics",str(Path(d,"missing.json"))])
  self.assertEqual(code,2);self.assertEqual(out["status"],"unavailable")
 def test_document_data_numeric_commands(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);txt=root/"note.txt";txt.write_text("hello target",encoding="utf-8");csvp=root/"data.csv";csvp.write_text("team,value\na,1\na,3\n",encoding="utf-8")
   code,out=self.capture(["document","extract",str(txt),"--query","target"]);self.assertEqual(code,0);self.assertIn("target",out["content"])
   code,out=self.capture(["data","analyze",str(csvp),"--group-by","team"]);self.assertEqual(code,0);self.assertEqual(out["result"]["rows"],2)
   code,out=self.capture(["numeric","describe","1","2","3"]);self.assertEqual(code,0);self.assertEqual(out["result"]["median"],2)
 def test_research_is_explicitly_untrusted(self):
  class R:
   status_code=200;content=b"<title>x</title><p>public text</p>";text=content.decode()
  with patch("httpx.get",return_value=R()):code,out=self.capture(["research","fetch","https://example.com"])
  self.assertEqual(code,0);self.assertTrue(out["untrusted_content"])
if __name__=="__main__":unittest.main()
