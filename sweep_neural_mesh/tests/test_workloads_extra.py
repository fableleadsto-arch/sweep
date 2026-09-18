import tempfile,unittest
from pathlib import Path
from sweep_neural_mesh.workloads_extra import data_clean,data_forecast,document_compare,export_json
class ExtraWorkloadTests(unittest.TestCase):
 def test_document_compare_and_export(self):
  with tempfile.TemporaryDirectory() as d:
   a=Path(d,"a.txt");b=Path(d,"b.txt");a.write_text("one\ntwo\n",encoding="utf-8");b.write_text("one\nthree\n",encoding="utf-8");out=document_compare(str(a),str(b));self.assertTrue(out["changed"]);target=Path(d,"out.json");saved=export_json(out,str(target));self.assertEqual(saved["status"],"completed");self.assertTrue(target.exists())
 def test_clean_and_forecast(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d,"data.csv");p.write_text("x,y\n1,2\n1,2\n2,4\n3,6\n",encoding="utf-8");clean=data_clean(str(p),fill="zero");self.assertEqual(clean["duplicates_removed"],1);forecast=data_forecast(str(p),"y",2);self.assertEqual(len(forecast["predictions"]),2);self.assertGreater(forecast["slope"],0)
if __name__=="__main__":unittest.main()
