"""Topological, fail-closed execution for MeshGraph."""
from __future__ import annotations
import logging,time
from typing import Any
from .graph import MeshGraph
from .node import Modality,NodeResult,NodeStatus
from .packet import NeuralPacket
logger=logging.getLogger(__name__)
class ExecutionResult:
 def __init__(self,graph_id=""):
  self.graph_id=graph_id;self.success=True;self.node_results={};self.output_packets=[];self.total_latency_ms=0.0;self.total_nodes_executed=0;self.total_nodes_failed=0;self.total_nodes_blocked=0;self.warnings=[];self.metadata={}
 def to_dict(self):
  return {"graph_id":self.graph_id,"success":self.success,"total_latency_ms":self.total_latency_ms,"nodes_executed":self.total_nodes_executed,"nodes_failed":self.total_nodes_failed,"nodes_blocked":self.total_nodes_blocked,"output_packets":len(self.output_packets),"warnings":self.warnings}
class ExecutionEngine:
 def __init__(self):self._execution_history=[]
 def execute(self,graph:MeshGraph,initial_inputs:dict[str,NeuralPacket]|None=None)->ExecutionResult:
  t0=time.perf_counter();result=ExecutionResult(graph.graph_id);packet_map=dict(initial_inputs or {});nodes={n.node_id:n for n in graph.nodes}
  try:order=graph.execution_order
  except ValueError as exc:
   result.success=False;result.warnings.append(str(exc));result.total_latency_ms=(time.perf_counter()-t0)*1000;self._execution_history.append(result);return result
  for node_id in order:
   node=nodes.get(node_id)
   if node is None:
    failure=NodeResult(success=False,error=f"Graph execution order references missing node {node_id}",metadata={"failure_kind":"invalid_graph"});result.node_results[node_id]=failure;result.total_nodes_failed+=1;result.warnings.append(failure.error or "invalid graph");continue
   predecessors=graph.predecessors(node_id);missing=[p.node_id for p in predecessors if p.node_id not in packet_map]
   if missing:
    node.status=NodeStatus.FAILED;blocked=NodeResult(success=False,error=f"Blocked by failed or missing predecessors: {', '.join(missing)}",metadata={"failure_kind":"dependency_unavailable","blocked_by":missing});graph.store_result(node_id,blocked);result.node_results[node_id]=blocked;result.total_nodes_blocked+=1;result.warnings.append(f"Node {node.name} ({node_id}) blocked: {blocked.error}");continue
   input_packets=[packet_map[p.node_id] for p in predecessors] if predecessors else ([packet_map[node_id]] if node_id in packet_map else [])
   if input_packets:
    primary=input_packets[0];node_result=node.execute(primary.data,packets=input_packets,metadata=primary.metadata)
   else:node_result=node.execute()
   graph.store_result(node_id,node_result);result.node_results[node_id]=node_result;result.total_nodes_executed+=1
   if node_result.success:
    metadata={**node_result.metadata,"latency_ms":node_result.latency_ms,"input_packet_ids":[p.packet_id for p in input_packets],"confidence_status":"reported" if node_result.confidence>0 else "unspecified"}
    packet_map[node_id]=NeuralPacket(data=node_result.output,modality=node.schema.output_modalities[0] if node.schema.output_modalities else Modality.TENSOR,confidence=node_result.confidence,source_node_id=node_id,source_node_name=node.name,parent_packet_id=input_packets[0].packet_id if input_packets else None,metadata=metadata)
   else:
    result.total_nodes_failed+=1;result.warnings.append(f"Node {node.name} ({node_id}) failed: {node_result.error}");logger.warning("Node %s failed: %s",node.name,node_result.error)
  for leaf in graph.leaves():
   if leaf.node_id in packet_map:result.output_packets.append(packet_map[leaf.node_id])
  result.success=result.total_nodes_failed==0 and result.total_nodes_blocked==0;result.total_latency_ms=(time.perf_counter()-t0)*1000;self._execution_history.append(result);return result
 @property
 def history(self):return list(self._execution_history)
 @property
 def stats(self):
  if not self._execution_history:return {"executions":0}
  total=len(self._execution_history);success=sum(1 for r in self._execution_history if r.success)
  return {"executions":total,"success_rate":success/total,"avg_latency_ms":sum(r.total_latency_ms for r in self._execution_history)/total,"nodes_blocked":sum(r.total_nodes_blocked for r in self._execution_history)}
 def __repr__(self):return f"ExecutionEngine(history={len(self._execution_history)})"
