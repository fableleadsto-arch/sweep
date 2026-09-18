"""Adapters from the terminal router to SWEEP's existing production engines."""
from __future__ import annotations
from dataclasses import asdict,is_dataclass
from typing import Any
def query_neural(question:str)->dict[str,Any]:
 try:
  from sweep_neural_mesh.sweep_api import SweepAPI
  result=SweepAPI().query(question);method=str(getattr(result,"method","error"));confidence=float(getattr(result,"confidence",0.0));answer=str(getattr(result,"answer",""))
  if method in {"error","unavailable"} or confidence<=0.0:return {"status":"unavailable","method":method,"confidence":confidence,"reason":answer or "No production inference provider produced a result."}
  return {"status":"completed","method":method,"confidence":confidence,"observation":answer,"reasoning":str(getattr(result,"reasoning","")),"components":list(getattr(result,"components",[]) or [])}
 except Exception as exc:return {"status":"unavailable","method":"exception","confidence":0.0,"reason":f"SweepAPI unavailable ({type(exc).__name__})"}
def device_plan(message:str)->dict[str,Any]:
 try:
  from companion.device_intents import plan_device_actions
  p=plan_device_actions(message);return {"status":"planned","recognized":p.recognized,"actions":[asdict(a) if is_dataclass(a) else dict(a.__dict__) for a in p.actions],"clarification":p.clarification,"confidence":p.confidence,"execution":"not_performed; submit through approval-backed Device Host"}
 except Exception as exc:return {"status":"unavailable","reason":f"Device planner unavailable ({type(exc).__name__})","execution":"not_performed"}
