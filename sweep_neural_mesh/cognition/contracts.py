"""Versioned structured contracts for SWEEP cognition."""
from __future__ import annotations
import time,uuid
from dataclasses import asdict,dataclass,field
from typing import Any
CONTRACT_VERSION="sweep-cognition/1"
def _id(prefix):return f"{prefix}_{uuid.uuid4().hex[:16]}"
def _confidence(value):return max(0.0,min(1.0,float(value)))
@dataclass
class Observation:
 source:str;data:Any;modality:str="structured";confidence:float=0.0;id:str=field(default_factory=lambda:_id("obs"));timestamp:float=field(default_factory=time.time)
 def __post_init__(self):self.confidence=_confidence(self.confidence)
 def to_dict(self):return {"version":CONTRACT_VERSION,**asdict(self)}
@dataclass
class Evidence:
 observation_ids:list[str];source_key:str;reliability:float=0.0;id:str=field(default_factory=lambda:_id("ev"))
 def __post_init__(self):self.reliability=_confidence(self.reliability)
 def to_dict(self):return {"version":CONTRACT_VERSION,**asdict(self)}
@dataclass
class Claim:
 text:str;status:str="proposed";confidence:float=0.0;evidence_ids:list[str]=field(default_factory=list);id:str=field(default_factory=lambda:_id("claim"))
 def __post_init__(self):self.confidence=_confidence(self.confidence)
 def to_dict(self):return {"version":CONTRACT_VERSION,**asdict(self)}
@dataclass
class Hypothesis:
 text:str;alternatives:list[str]=field(default_factory=list);status:str="untested";id:str=field(default_factory=lambda:_id("hyp"))
 def to_dict(self):return {"version":CONTRACT_VERSION,**asdict(self)}
@dataclass
class ReasoningEvent:
 stage:str;summary:str;status:str="completed";input_ids:list[str]=field(default_factory=list);output_ids:list[str]=field(default_factory=list);id:str=field(default_factory=lambda:_id("reason"));timestamp:float=field(default_factory=time.time)
 def to_dict(self):return {"version":CONTRACT_VERSION,**asdict(self)}
@dataclass
class Conclusion:
 answer:str;status:str;confidence:float;claim_ids:list[str]=field(default_factory=list);evidence_ids:list[str]=field(default_factory=list);conflicts:list[str]=field(default_factory=list);uncertainty:list[str]=field(default_factory=list);failure_modes:list[str]=field(default_factory=list);event_ids:list[str]=field(default_factory=list);id:str=field(default_factory=lambda:_id("conclusion"));timestamp:float=field(default_factory=time.time)
 def __post_init__(self):self.confidence=_confidence(self.confidence)
 def to_dict(self):return {"version":CONTRACT_VERSION,**asdict(self)}
