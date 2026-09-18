import tempfile,unittest
from pathlib import Path
from sweep_neural_mesh.specialists.generative import LocalCausalLMAdapter,ModelSpec,default_qwen_specs
class QwenSpecialistTests(unittest.TestCase):
 def test_missing_model_is_honest(self):
  p=LocalCausalLMAdapter(ModelSpec("q","Q/q","/definitely/missing","general",("generation",))).probe();self.assertEqual(p.status,"missing")
 def test_local_files_are_probed_without_loading(self):
  with tempfile.TemporaryDirectory() as d:
   Path(d,"config.json").write_text("{}",encoding="utf-8");Path(d,"model.safetensors").write_bytes(b"x");p=LocalCausalLMAdapter(ModelSpec("q","Q/q",d,"general",("generation",))).probe();self.assertEqual(p.status,"available_untested");self.assertTrue(p.has_weights)
 def test_multimodal_profile_stays_blocked(self):
  self.assertEqual(LocalCausalLMAdapter(default_qwen_specs()["multimodal"]).probe().status,"blocked")
 def test_node_is_inside_mesh_contract(self):
  adapter=LocalCausalLMAdapter(ModelSpec("q","Q/q","unused","general",("language_generation",)));adapter.generate=lambda bundle,prompt,system="",max_new_tokens=None:{"text":"answer","confidence":0.0,"verification_status":"unverified"};result=adapter.build_node({}).execute("question");self.assertTrue(result.success);self.assertEqual(result.confidence,0.0);self.assertEqual(result.output["verification_status"],"unverified")
if __name__=="__main__":unittest.main()
