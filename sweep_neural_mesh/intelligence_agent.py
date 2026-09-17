"""Terminal intelligence agent backed by SWEEP's trained task router.

Run: python -m sweep_neural_mesh.intelligence_agent
One shot: python -m sweep_neural_mesh.intelligence_agent --once "extract names from ..."
External/OS actions are never reported complete without a real observation.
"""
from __future__ import annotations
import argparse, json, math, re, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
VERSION="sweep-terminal-agent/1"
TOKEN_RE=re.compile(r"[a-z0-9][a-z0-9_-]{1,}")
DEFAULT_MODEL=Path(__file__).parent/"training"/"artifacts"/"intelligence_router.json"
DEFAULT_MEMORY=Path.home()/".sweep"/"memory.json"
def tokenize(text):
 words=TOKEN_RE.findall(text.lower()); return words+[f"{a}::{b}" for a,b in zip(words,words[1:])]
class IntelligenceRouter:
 def __init__(self,path=DEFAULT_MODEL):
  self.path=Path(path)
  if not self.path.exists():
   from sweep_neural_mesh.training.train_intelligence_router import train
   train(self.path.parent)
  self.artifact=json.loads(self.path.read_text(encoding="utf-8"))
  if self.artifact.get("version")!="sweep-intelligence-router/1": raise ValueError("unsupported intelligence router artifact")
 def classify(self,text):
  a=self.artifact; toks=Counter(tokenize(text)); vs=max(1,len(a["vocabulary"])); total=sum(a["class_documents"].values()); raw={}
  for label in a["labels"]:
   prior=(a["class_documents"][label]+1)/(total+len(a["labels"])); counts=a["token_counts"][label]; denom=sum(counts.values())+vs
   raw[label]=math.log(prior)+sum(n*math.log((counts.get(tok,0)+1)/denom) for tok,n in toks.items())
  peak=max(raw.values()); probs={k:math.exp(v-peak) for k,v in raw.items()}; z=sum(probs.values()); probs={k:v/z for k,v in probs.items()}; label=max(probs,key=probs.get)
  return {"intent":label,"confidence":probs[label],"probabilities":dict(sorted(probs.items(),key=lambda kv:kv[1],reverse=True)[:3]),"model_version":a["version"]}
class CompatibleFileMemory:
 """Reads/writes companion.memory's version-1 JSON shape."""
 def __init__(self,path): self.path=Path(path)
 def _load(self):
  try:return json.loads(self.path.read_text(encoding="utf-8"))
  except (OSError,ValueError):return {"version":1,"memories":[]}
 def _save(self,data):self.path.parent.mkdir(parents=True,exist_ok=True);self.path.write_text(json.dumps(data,indent=2),encoding="utf-8")
 def remember(self,content,user_id="terminal"):
  content=" ".join(content.split())
  if not content:raise ValueError("memory content cannot be empty")
  data=self._load();now=datetime.now(timezone.utc).isoformat();entries=data.setdefault("memories",[])
  for item in entries:
   if item.get("userId")==user_id and item.get("content")==content:item["updatedAt"]=now;self._save(data);return item
  item={"id":f"{user_id}:{int(time.time()*1000)}","userId":user_id,"workspaceId":None,"content":content,"kind":"context","tags":["terminal"],"source":VERSION,"confidence":0.8,"createdAt":now,"updatedAt":now,"expiresAt":None};entries.append(item);self._save(data);return item
 def recall(self,query,user_id="terminal",limit=5):
  terms=set(TOKEN_RE.findall(query.lower()));rows=[x for x in self._load().get("memories",[]) if x.get("userId")==user_id];rows.sort(key=lambda x:(len(terms&set(TOKEN_RE.findall(x.get("content","").lower()))),x.get("updatedAt","")),reverse=True);return rows[:limit]
def extract(text):return {"emails":sorted(set(re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",text))),"urls":sorted(set(re.findall(r"https?://[^\s]+",text))),"dates":sorted(set(re.findall(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",text))),"capitalized_entities":sorted(set(re.findall(r"\b[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*)*",text)))}
def summarize(text,limit=3):return " ".join([s.strip() for s in re.split(r"(?<=[.!?])\s+",text) if s.strip()][:limit])
def plan(text,routing):
 intent=routing["intent"]
 if intent=="extraction":result={"status":"completed","observation":extract(text)}
 elif intent=="summarization":result={"status":"completed","observation":summarize(text)}
 elif intent=="device_action":result={"status":"unavailable","reason":"Device Host approval/execution is not connected; no OS action was performed."}
 else:result={"status":"planned","next_steps":["define scope and authorization","collect traceable sources","separate observations from claims","check contradictions and provenance","report confidence and limitations"]}
 return {"version":VERSION,"request":text,"routing":routing,"safety_boundary":"Only lawful, authorized, public-source intelligence work is allowed.","result":result}
def main(argv=None):
 p=argparse.ArgumentParser(description="SWEEP terminal intelligence agent");p.add_argument("--model",type=Path,default=DEFAULT_MODEL);p.add_argument("--memory-file",type=Path,default=DEFAULT_MEMORY);p.add_argument("--once");p.add_argument("--remember");p.add_argument("--recall");args=p.parse_args(argv);router=IntelligenceRouter(p.parse_args(argv).model);memory=CompatibleFileMemory(args.memory_file)
 if args.remember is not None:print(json.dumps({"remembered":memory.remember(args.remember)},indent=2));return 0
 if args.recall is not None:print(json.dumps({"memories":memory.recall(args.recall)},indent=2));return 0
 def run(text):
  out=plan(text,router.classify(text));memory.remember(f"User requested: {text}");return out
 if args.once is not None:print(json.dumps(run(args.once),indent=2));return 0
 print(f"SWEEP Intelligence Agent {VERSION}. Commands: /remember TEXT, /recall QUERY, /quit")
 while True:
  try:text=input("sweep> ").strip()
  except (EOFError,KeyboardInterrupt):print();return 0
  if not text:continue
  if text in {"/quit","/exit"}:return 0
  if text.startswith("/remember "):print(json.dumps(memory.remember(text[10:]),indent=2));continue
  if text.startswith("/recall "):print(json.dumps(memory.recall(text[8:]),indent=2));continue
  print(json.dumps(run(text),indent=2))
if __name__=="__main__":raise SystemExit(main())
