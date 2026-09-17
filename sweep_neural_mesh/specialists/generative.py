"""Replaceable local-only causal-LM specialists for the SWEEP mesh."""
from __future__ import annotations
import os,time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from sweep_neural_mesh.core.node import Framework,Modality,NeuralNode,NodeSchema,NodeVersion
@dataclass(frozen=True)
class ModelSpec:
 name:str;checkpoint:str;local_path:str;role:str;capabilities:tuple[str,...];license:str="unverified";blocked_reason:str="";max_new_tokens:int=256
@dataclass(frozen=True)
class ModelProbe:
 name:str;status:str;path:str;has_config:bool=False;has_weights:bool=False;reason:str=""
 def to_dict(self):return self.__dict__.copy()
def default_qwen_specs():
 return {
  "general":ModelSpec("qwen2.5-0.5b-instruct","Qwen/Qwen2.5-0.5B-Instruct",os.getenv("SWEEP_QWEN_GENERAL_PATH","models/generative/qwen2.5-0.5b-instruct"),"general",("language_generation","teaching","creative_writing","planning"),"verify-upstream"),
  "coding":ModelSpec("qwen2.5-coder-1.5b-instruct","Qwen/Qwen2.5-Coder-1.5B-Instruct",os.getenv("SWEEP_QWEN_CODER_PATH","models/generative/qwen2.5-coder-1.5b-instruct"),"coding",("code_generation","code_explanation","debugging","refactoring"),"verify-upstream"),
  "math":ModelSpec("qwen2.5-math-1.5b-instruct","Qwen/Qwen2.5-Math-1.5B-Instruct",os.getenv("SWEEP_QWEN_MATH_PATH","models/generative/qwen2.5-math-1.5b-instruct"),"math",("mathematical_reasoning","formula_derivation","problem_solving"),"verify-upstream"),
  "multimodal":ModelSpec("qwen2.5-vl-3b-instruct","Qwen/Qwen2.5-VL-3B-Instruct",os.getenv("SWEEP_QWEN_VL_PATH","models/multimodal/qwen2.5-vl-3b-instruct"),"multimodal",("image_understanding","diagram_reading","document_image_analysis"),"verify-upstream","Requires a dedicated vision adapter, installed weights, adequate hardware, and held-out multimodal evaluation."),
 }
class LocalCausalLMAdapter:
 framework=Framework.PYTORCH
 def __init__(self,spec:ModelSpec):self.spec=spec
 def probe(self):
  path=Path(self.spec.local_path)
  if self.spec.blocked_reason:return ModelProbe(self.spec.name,"blocked",str(path),reason=self.spec.blocked_reason)
  if not path.is_dir():return ModelProbe(self.spec.name,"missing",str(path),reason="local model directory not found")
  config=(path/"config.json").is_file();weights=any(path.glob("*.safetensors")) or any(path.glob("*.bin"));status="available_untested" if config and weights else "invalid"
  return ModelProbe(self.spec.name,status,str(path),config,weights,"" if status=="available_untested" else "config or weights missing")
 def load(self):
  probe=self.probe()
  if probe.status!="available_untested":raise RuntimeError(f"{self.spec.name} unavailable: {probe.reason}")
  import torch
  from transformers import AutoModelForCausalLM,AutoTokenizer
  tokenizer=AutoTokenizer.from_pretrained(self.spec.local_path,local_files_only=True,trust_remote_code=False)
  model=AutoModelForCausalLM.from_pretrained(self.spec.local_path,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float32,low_cpu_mem_usage=True);model.eval();return {"model":model,"tokenizer":tokenizer,"spec":self.spec}
 def generate(self,bundle:dict[str,Any],prompt:str,system:str="",max_new_tokens:int|None=None):
  import torch
  model=bundle["model"];tokenizer=bundle["tokenizer"];messages=[]
  if system:messages.append({"role":"system","content":system})
  messages.append({"role":"user","content":prompt});text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);inputs=tokenizer(text,return_tensors="pt");started=time.perf_counter()
  with torch.no_grad():output=model.generate(**inputs,max_new_tokens=min(max_new_tokens or self.spec.max_new_tokens,self.spec.max_new_tokens),do_sample=False,pad_token_id=tokenizer.eos_token_id)
  answer=tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True).strip();return {"text":answer,"model":self.spec.name,"role":self.spec.role,"confidence":0.0,"verification_status":"unverified","latency_ms":(time.perf_counter()-started)*1000}
 def build_node(self,bundle):
  def execute(data,**kwargs):
   request=data if isinstance(data,dict) else {"prompt":str(data)};return self.generate(bundle,str(request.get("prompt","")),str(request.get("system","")),request.get("max_new_tokens"))
  return NeuralNode(name=self.spec.name,framework=self.framework,execute_fn=execute,capabilities=list(self.spec.capabilities),schema=NodeSchema(input_modalities=[Modality.TEXT],output_modalities=[Modality.STRUCTURED],input_dtype="utf8",output_dtype="json"),version=NodeVersion(model_id=self.spec.checkpoint,weights_version="local",architecture_version="hf-causal-lm/1"),tags={"role":self.spec.role,"loading":"local_only","verification":"required"})
