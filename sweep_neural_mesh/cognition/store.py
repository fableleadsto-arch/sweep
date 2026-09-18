"""Append-only reasoning-event and conclusion persistence."""
from __future__ import annotations
import json
from pathlib import Path
from .contracts import Conclusion,ReasoningEvent
class ReasoningStore:
 def __init__(self,path):self.path=Path(path)
 def _append(self,record):
  self.path.parent.mkdir(parents=True,exist_ok=True)
  with self.path.open("a",encoding="utf-8") as f:f.write(json.dumps(record,sort_keys=True)+"\n")
 def persist(self,events:list[ReasoningEvent],conclusion:Conclusion):
  for event in events:self._append({"record_type":"reasoning_event",**event.to_dict()})
  self._append({"record_type":"conclusion",**conclusion.to_dict()})
 def explain_conclusion(self,conclusion_id):
  events={};conclusion=None
  if not self.path.exists():return None
  for line in self.path.read_text(encoding="utf-8").splitlines():
   try:record=json.loads(line)
   except ValueError:continue
   if record.get("record_type")=="reasoning_event":events[record["id"]]=record
   elif record.get("record_type")=="conclusion" and record.get("id")==conclusion_id:conclusion=record
  if conclusion is None:return None
  return {"version":conclusion["version"],"conclusion":conclusion,"events":[events[e] for e in conclusion.get("event_ids",[]) if e in events]}
