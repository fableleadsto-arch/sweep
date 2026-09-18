"""Trained local intent router and deterministic terminal task helpers."""
from __future__ import annotations
import argparse,json,math,re
from collections import Counter
from pathlib import Path
VERSION="sweep-terminal-agent/2";TOKEN_RE=re.compile(r"[a-z0-9][a-z0-9_-]{1,}");DEFAULT_MODEL=Path(__file__).parent/"training"/"artifacts"/"intelligence_router.json";DEFAULT_MEMORY=Path.home()/".relayhub"/"relai-memory.json"
def tokenize(text):
 words=TOKEN_RE.findall(text.lower());return words+[f"{a}::{b}" for a,b in zip(words,words[1:])]
class IntelligenceRouter:
 def __init__(self,path=DEFAULT_MODEL):
  self.path=Path(path)
  if not self.path.exists():
   from sweep_neural_mesh.training.train_intelligence_router import train
   train(self.path.parent)
  self.artifact=json.loads(self.path.read_text(encoding="utf-8"))
  if self.artifact.get("version")!="sweep-intelligence-router/1":raise ValueError("unsupported intelligence router artifact")
 def classify(self,text):
  a=self.artifact;toks=Counter(tokenize(text));vs=max(1,len(a["vocabulary"]));total=sum(a["class_documents"].values());raw={}
  for label in a["labels"]:
   prior=(a["class_documents"][label]+1)/(total+len(a["labels"]));counts=a["token_counts"][label];denom=sum(counts.values())+vs;raw[label]=math.log(prior)+sum(n*math.log((counts.get(tok,0)+1)/denom) for tok,n in toks.items())
  peak=max(raw.values());probs={k:math.exp(v-peak) for k,v in raw.items()};z=sum(probs.values());probs={k:v/z for k,v in probs.items()};label=max(probs,key=probs.get);return {"intent":label,"confidence":probs[label],"probabilities":dict(sorted(probs.items(),key=lambda kv:kv[1],reverse=True)[:3]),"model_version":a["version"]}
from sweep_neural_mesh.terminal_memory import ProductionMemoryAdapter,CompatibleFileMemory
def extract(text):return {"emails":sorted(set(re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",text))),"urls":sorted(set(re.findall(r"https?://[^\s]+",text))),"dates":sorted(set(re.findall(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",text))),"capitalized_entities":sorted(set(re.findall(r"\b[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*)*",text)))}
def summarize(text,limit=3):return " ".join([s.strip() for s in re.split(r"(?<=[.!?])\s+",text) if s.strip()][:limit])
def plan(text,routing):
 intent=routing["intent"]
 if intent=="extraction":result={"status":"completed","observation":extract(text)}
 elif intent=="summarization":result={"status":"completed","observation":summarize(text)}
 elif intent=="device_action":result={"status":"unavailable","reason":"Submit through the approval-backed Device Host; no OS action was performed."}
 else:result={"status":"planned","next_steps":["define scope and authorization","collect traceable sources","separate observations from claims","check contradictions and provenance","report confidence and limitations"]}
 return {"version":VERSION,"request":text,"routing":routing,"safety_boundary":"Only lawful, authorized, public-source intelligence work is allowed.","result":result}
def main(argv=None):
 from sweep_neural_mesh.terminal import main as terminal_main
 return terminal_main(argv)
if __name__=="__main__":raise SystemExit(main())
