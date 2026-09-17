"""Environment-backed settings for the local Device Host HTTP surface."""
from __future__ import annotations
import json,os
from dataclasses import dataclass,field
_TRUE={"1","true","yes","on"}
def _bool(name,default=False):
 value=os.getenv(name);return default if value is None else value.strip().lower() in _TRUE
def _json(name,default):
 value=os.getenv(name)
 if not value:return default
 parsed=json.loads(value)
 if not isinstance(parsed,type(default)):raise ValueError(f"{name} must decode to {type(default).__name__}")
 return parsed
@dataclass(frozen=True)
class DeviceRouteSettings:
 enabled:bool=False;simulation_mode:bool=False;dry_run_default:bool=True
 permissions_file:str=".relayhub/device-host-permissions.json";audit_file:str=".relayhub/device-host-audit.jsonl"
 allowed_roots:list[str]=field(default_factory=list);path_aliases:dict[str,str]=field(default_factory=dict);app_allowlist:dict[str,list[str]]=field(default_factory=dict)
 @classmethod
 def from_env(cls):
  return cls(enabled=_bool("COMPANION_DEVICE_HOST_ENABLED",False),simulation_mode=_bool("COMPANION_DEVICE_HOST_SIMULATION",False),dry_run_default=_bool("COMPANION_DEVICE_HOST_DRY_RUN",True),permissions_file=os.getenv("COMPANION_DEVICE_HOST_PERMISSIONS_FILE",cls.permissions_file),audit_file=os.getenv("COMPANION_DEVICE_HOST_AUDIT_FILE",cls.audit_file),allowed_roots=[str(x) for x in _json("COMPANION_DEVICE_HOST_ALLOWED_ROOTS",[])],path_aliases={str(k):str(v) for k,v in _json("COMPANION_DEVICE_HOST_PATH_ALIASES",{}).items()},app_allowlist={str(k).lower():[str(x) for x in v] for k,v in _json("COMPANION_DEVICE_HOST_APP_ALLOWLIST",{}).items()})
