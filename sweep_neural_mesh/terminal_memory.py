"""Synchronous terminal adapter over companion.memory.MemoryService."""
from __future__ import annotations
import asyncio
from pathlib import Path
class ProductionMemoryAdapter:
 def __init__(self,path:Path,user_id="terminal",workspace_id=None):
  from companion.config import BrainSettings
  from companion.memory import MemoryService
  self.user_id=user_id;self.workspace_id=workspace_id;self.service=MemoryService(BrainSettings(memory_file=str(path)))
 @staticmethod
 def _run(awaitable):
  try:asyncio.get_running_loop()
  except RuntimeError:return asyncio.run(awaitable)
  raise RuntimeError("terminal memory sync adapter cannot run inside an active event loop")
 def remember(self,content,user_id=None):
  entry=self._run(self.service.remember(user_id or self.user_id,content,workspace_id=self.workspace_id,kind="context",tags=["terminal"],source="sweep-terminal-agent/2",confidence=0.8));return entry.to_dict()
 def recall(self,query,user_id=None,limit=5):
  entries=self._run(self.service.search(user_id or self.user_id,query,workspace_id=self.workspace_id,limit=limit));return [entry.to_dict() for entry in entries]
CompatibleFileMemory=ProductionMemoryAdapter
