"""Enforce training/evaluation boundaries from the external source lock."""
from __future__ import annotations
import json
from pathlib import Path
LOCK=Path(__file__).with_name("external_source_lock.json")
class SourcePolicy:
 def __init__(self,path:Path=LOCK):self.rows={x["name"]:x for x in json.loads(path.read_text())["sources"]}
 def require_training_allowed(self,name:str):
  row=self.rows[name]
  if row.get("usage")=="evaluation_only" or row.get("training_allowed") is False:raise PermissionError(f"{name} is evaluation-only and cannot be used for training")
  return row
 def evaluation_source(self,name:str):return self.rows[name]
