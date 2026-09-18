from __future__ import annotations
import unittest
from sweep_neural_mesh.core.engine import ExecutionEngine
from sweep_neural_mesh.core.graph import MeshGraph
from sweep_neural_mesh.core.node import NeuralNode
from sweep_neural_mesh.core.packet import NeuralPacket
class EngineReliabilityTests(unittest.TestCase):
 def test_failed_predecessor_blocks_descendant_without_execution(self):
  called=[]
  def fail(*args,**kwargs):raise RuntimeError("upstream failed")
  upstream=NeuralNode(node_id="upstream",name="upstream",execute_fn=fail)
  downstream=NeuralNode(node_id="downstream",name="downstream",execute_fn=lambda *a,**k:called.append(True))
  graph=MeshGraph();graph.add_node(upstream);graph.add_node(downstream);graph.add_edge(upstream,downstream)
  result=ExecutionEngine().execute(graph)
  self.assertFalse(result.success);self.assertEqual(called,[]);self.assertEqual(result.total_nodes_executed,1);self.assertEqual(result.total_nodes_failed,1);self.assertEqual(result.total_nodes_blocked,1);self.assertEqual(result.node_results["downstream"].metadata["failure_kind"],"dependency_unavailable")
 def test_unspecified_confidence_is_not_promoted(self):
  node=NeuralNode(node_id="node",name="node",execute_fn=lambda value,**kwargs:value*2);graph=MeshGraph();graph.add_node(node);source=NeuralPacket(data=3,confidence=0.0)
  result=ExecutionEngine().execute(graph,{"node":source});self.assertTrue(result.success);self.assertEqual(result.output_packets[0].confidence,0.0);self.assertEqual(result.output_packets[0].metadata["confidence_status"],"unspecified");self.assertEqual(result.output_packets[0].parent_packet_id,source.packet_id)
 def test_blocking_propagates_through_chain(self):
  calls=[];a=NeuralNode(node_id="a",execute_fn=lambda:(_ for _ in ()).throw(ValueError("fail")));b=NeuralNode(node_id="b",execute_fn=lambda *x,**k:calls.append("b"));c=NeuralNode(node_id="c",execute_fn=lambda *x,**k:calls.append("c"));g=MeshGraph()
  for n in (a,b,c):g.add_node(n)
  g.add_edge(a,b);g.add_edge(b,c);r=ExecutionEngine().execute(g);self.assertEqual(calls,[]);self.assertEqual(r.total_nodes_blocked,2)
if __name__=="__main__":unittest.main()
