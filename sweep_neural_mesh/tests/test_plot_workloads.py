import tempfile,unittest
from pathlib import Path
from sweep_neural_mesh.plot_workloads import data_plot
class PlotWorkloadTests(unittest.TestCase):
 def test_plot_is_real_or_explicitly_unavailable(self):
  with tempfile.TemporaryDirectory() as d:
   source=Path(d,"data.csv");target=Path(d,"chart.png");source.write_text("x,y\n1,2\n2,4\n3,3\n",encoding="utf-8")
   out=data_plot(str(source),"line","x","y",str(target))
   self.assertIn(out["status"],{"completed","unavailable"})
   if out["status"]=="completed":self.assertTrue(target.exists());self.assertGreater(target.stat().st_size,8);self.assertEqual(len(out["output"]["sha256"]),64)
 def test_output_is_required(self):
  with tempfile.TemporaryDirectory() as d:
   source=Path(d,"data.csv");source.write_text("x,y\n1,2\n",encoding="utf-8")
   out=data_plot(str(source));self.assertEqual(out["status"],"error");self.assertIn("output path",out["reason"])
if __name__=="__main__":unittest.main()
