"""Orchestrated SWEEP terminal. Neural delegation is opt-in; device actions are plan-only."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from sweep_neural_mesh.intelligence_agent import CompatibleFileMemory,DEFAULT_MEMORY,DEFAULT_MODEL,IntelligenceRouter,plan
from sweep_neural_mesh.terminal_orchestration import device_plan,query_neural
def execute(text,router,neural=False,neural_executor=query_neural,device_executor=device_plan):
 routing=router.classify(text)
 if routing["intent"]=="device_action":result=device_executor(text)
 elif neural and routing["intent"] not in {"extraction","summarization"}:result=neural_executor(text)
 else:return plan(text,routing)
 return {"version":"sweep-terminal-orchestrator/1","request":text,"routing":routing,"result":result}
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--model",type=Path,default=DEFAULT_MODEL);p.add_argument("--memory-file",type=Path,default=DEFAULT_MEMORY);p.add_argument("--neural",action="store_true");p.add_argument("--once");a=p.parse_args(argv);router=IntelligenceRouter(a.model);memory=CompatibleFileMemory(a.memory_file)
 def run(text):
  out=execute(text,router,a.neural);memory.remember(f"User requested: {text}");return out
 if a.once is not None:print(json.dumps(run(a.once),indent=2));return 0
 print("SWEEP orchestrated terminal. /quit to exit.")
 while True:
  try:text=input("sweep> ").strip()
  except (EOFError,KeyboardInterrupt):print();return 0
  if text in {"/quit","/exit"}:return 0
  if text:print(json.dumps(run(text),indent=2))
if __name__=="__main__":raise SystemExit(main())
