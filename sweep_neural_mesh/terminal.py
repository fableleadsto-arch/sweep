"""Orchestrated SWEEP terminal; device actions remain plan-only."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from sweep_neural_mesh.intelligence_agent import DEFAULT_MEMORY,DEFAULT_MODEL,IntelligenceRouter,plan
from sweep_neural_mesh.terminal_memory import ProductionMemoryAdapter
from sweep_neural_mesh.terminal_orchestration import device_plan,query_neural
def execute(text,router,neural=False,neural_executor=query_neural,device_executor=device_plan):
 routing=router.classify(text)
 if routing["intent"]=="device_action":result=device_executor(text)
 elif neural and routing["intent"] not in {"extraction","summarization"}:result=neural_executor(text)
 else:return plan(text,routing)
 return {"version":"sweep-terminal-orchestrator/2","request":text,"routing":routing,"result":result}
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--model",type=Path,default=DEFAULT_MODEL);p.add_argument("--memory-file",type=Path,default=DEFAULT_MEMORY);p.add_argument("--user-id",default="terminal");p.add_argument("--workspace-id");p.add_argument("--neural",action="store_true");p.add_argument("--once");p.add_argument("--remember");p.add_argument("--recall");p.add_argument("--status",action="store_true");a=p.parse_args(argv);router=IntelligenceRouter(a.model);memory=ProductionMemoryAdapter(a.memory_file,a.user_id,a.workspace_id)
 if a.status:print(json.dumps({"version":"sweep-terminal-orchestrator/2","neural_enabled":a.neural,"memory_backend":"companion.memory.MemoryService","device_execution":"plan_only","user_id":a.user_id,"workspace_id":a.workspace_id},indent=2));return 0
 if a.remember is not None:print(json.dumps({"remembered":memory.remember(a.remember)},indent=2));return 0
 if a.recall is not None:print(json.dumps({"memories":memory.recall(a.recall)},indent=2));return 0
 def run(text):out=execute(text,router,a.neural);memory.remember(f"User requested: {text}");return out
 if a.once is not None:print(json.dumps(run(a.once),indent=2));return 0
 print("SWEEP orchestrated terminal. Commands: /status, /remember TEXT, /recall QUERY, /quit")
 while True:
  try:text=input("sweep> ").strip()
  except (EOFError,KeyboardInterrupt):print();return 0
  if text in {"/quit","/exit"}:return 0
  if text=="/status":print(json.dumps({"neural_enabled":a.neural,"device_execution":"plan_only","memory_backend":"production"},indent=2));continue
  if text.startswith("/remember "):print(json.dumps(memory.remember(text[10:]),indent=2));continue
  if text.startswith("/recall "):print(json.dumps(memory.recall(text[8:]),indent=2));continue
  if text:print(json.dumps(run(text),indent=2))
if __name__=="__main__":raise SystemExit(main())
