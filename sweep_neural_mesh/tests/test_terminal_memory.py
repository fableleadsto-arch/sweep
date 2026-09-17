import sys,types,unittest
from pathlib import Path
class Entry:
 def __init__(self,content):self.content=content
 def to_dict(self):return {"content":self.content}
class Settings:
 def __init__(self,**kwargs):self.kwargs=kwargs
class Service:
 def __init__(self,settings):self.settings=settings
 async def remember(self,user_id,content,**kwargs):self.remembered=(user_id,content,kwargs);return Entry(content)
 async def search(self,user_id,query,**kwargs):self.searched=(user_id,query,kwargs);return [Entry("matched")]
config=types.ModuleType("companion.config");config.BrainSettings=Settings;memory=types.ModuleType("companion.memory");memory.MemoryService=Service
sys.modules["companion.config"]=config;sys.modules["companion.memory"]=memory
from sweep_neural_mesh.terminal_memory import ProductionMemoryAdapter
class TerminalMemoryTests(unittest.TestCase):
 def test_uses_production_service_for_remember_and_recall(self):
  adapter=ProductionMemoryAdapter(Path("memory.json"),"alice","workspace");self.assertEqual(adapter.remember("fact")["content"],"fact");self.assertEqual(adapter.recall("query"),[{"content":"matched"}]);self.assertEqual(adapter.service.remembered[0],"alice");self.assertEqual(adapter.service.searched[2]["workspace_id"],"workspace")
if __name__=="__main__":unittest.main()
