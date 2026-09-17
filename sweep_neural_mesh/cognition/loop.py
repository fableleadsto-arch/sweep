"""Bounded human-inspired cognitive workflow over SWEEP specialists and verifiers."""
from __future__ import annotations
from typing import Callable
from .contracts import Claim,Conclusion,Hypothesis,Observation,ReasoningEvent
from .store import ReasoningStore
class CognitiveLoop:
 def __init__(self,router:Callable,specialist:Callable,verifier:Callable,store:ReasoningStore,max_events=8):
  if not 4<=max_events<=8:raise ValueError("max_events must be between 4 and 8")
  self.router=router;self.specialist=specialist;self.verifier=verifier;self.store=store;self.max_events=max_events
 def run(self,request:str,observations:list[Observation]|None=None):
  observations=list(observations or []);events=[]
  def event(stage,summary,input_ids=None,output_ids=None,status="completed"):
   if len(events)>=self.max_events:raise RuntimeError("cognitive event budget exhausted")
   events.append(ReasoningEvent(stage,summary,status,input_ids or [],output_ids or []))
  event("perceive",f"Accepted {len(observations)} explicit observation(s).",output_ids=[o.id for o in observations])
  route=dict(self.router(request));event("frame",f"Routed request to domain={route.get('domain','unknown')} task={route.get('task','unknown')}.")
  generated=self.specialist({"request":request,"route":route,"observations":[o.to_dict() for o in observations]});candidate=str(generated.get("text",generated.get("answer",""))).strip();hypothesis=Hypothesis(candidate or "No candidate generated");event("hypothesize","Generated an untrusted candidate for verification.",output_ids=[hypothesis.id],status="completed" if candidate else "failed")
  verification=dict(self.verifier(request,hypothesis,observations));evidence_ids=[str(x) for x in verification.get("evidence_ids",[])];independent=max(0,int(verification.get("independent_sources",0)));conflicts=[str(x) for x in verification.get("conflicts",[])];verified=bool(verification.get("verified")) and bool(evidence_ids);confidence=max(0.0,min(1.0,float(verification.get("confidence",0.0)))) if verified else 0.0
  if independent==0:confidence=0.0
  elif independent==1:confidence=min(confidence,0.65)
  status="supported" if verified and not conflicts else "unresolved";claim=Claim(hypothesis.text,status,confidence,evidence_ids);event("test",f"Verifier returned {len(evidence_ids)} evidence item(s) from {independent} independent source(s).",[hypothesis.id,*[o.id for o in observations]],[claim.id,*evidence_ids],"completed" if verified else "inconclusive")
  event("critique",f"Recorded {len(conflicts)} conflict(s) and {len(verification.get('failure_modes',[]))} failure mode(s).",[claim.id]);uncertainty=[str(x) for x in verification.get("uncertainty",[])]
  if not verified:uncertainty.append("No evidence-backed verification was produced.")
  event("revise","Kept the claim supported or unresolved based on evidence and conflicts.",[claim.id],[claim.id]);answer=claim.text if status=="supported" else "Unable to verify a reliable conclusion from the available evidence.";conclusion=Conclusion(answer,status,confidence,[claim.id],evidence_ids,conflicts,uncertainty,[str(x) for x in verification.get("failure_modes",[])])
  event("conclude",f"Produced a {status} conclusion with confidence {confidence:.3f}.",[claim.id],[conclusion.id]);conclusion.event_ids=[e.id for e in events];self.store.persist(events,conclusion);return conclusion
 def explain_conclusion(self,conclusion_id):return self.store.explain_conclusion(conclusion_id)
