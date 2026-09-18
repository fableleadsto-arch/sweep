"""Unified multipurpose SWEEP command-line interface."""
from __future__ import annotations
import argparse,json,platform,sys
from pathlib import Path
VERSION="sweep-cli/5"
DEFAULT_REASONING=Path.home()/".relayhub"/"sweep-reasoning.jsonl"
DEFAULT_COGNITIVE_ARTIFACT=Path.home()/".relayhub"/"models"/"cognitive-router"
def emit(value,pretty=True):print(json.dumps(value,indent=2 if pretty else None,default=str,sort_keys=pretty))
def execute(text,router,neural=False,neural_executor=None,device_executor=None):
 from sweep_neural_mesh.intelligence_agent import plan
 from sweep_neural_mesh.terminal_orchestration import device_plan,query_neural
 neural_executor=neural_executor or query_neural;device_executor=device_executor or device_plan;routing=router.classify(text)
 if routing["intent"]=="device_action":result=device_executor(text)
 elif neural and routing["intent"] not in {"extraction","summarization"}:result=neural_executor(text)
 else:return plan(text,routing)
 return {"version":VERSION,"request":text,"routing":routing,"result":result}
def model_status():
 from sweep_neural_mesh.specialists import LocalCausalLMAdapter,default_qwen_specs
 return {name:LocalCausalLMAdapter(spec).probe().to_dict() for name,spec in default_qwen_specs().items()}
def status_payload():
 from sweep_neural_mesh.intelligence_agent import DEFAULT_MODEL
 metrics=Path("docs/metrics/cognitive_router_metrics.json")
 return {"version":VERSION,"python":platform.python_version(),"platform":platform.platform(),"terminal_router":{"artifact":str(DEFAULT_MODEL),"available":DEFAULT_MODEL.exists()},"cognitive_router":{"metrics":str(metrics),"metrics_available":metrics.exists(),"artifact":str(DEFAULT_COGNITIVE_ARTIFACT/"cognitive_router.pkl"),"artifact_available":(DEFAULT_COGNITIVE_ARTIFACT/"cognitive_router.pkl").exists()},"reasoning_store":str(DEFAULT_REASONING),"models":model_status(),"device_host":"permission_gated","arbitrary_shell":False}
def memory_adapter(args):
 from sweep_neural_mesh.terminal_memory import ProductionMemoryAdapter
 return ProductionMemoryAdapter(args.memory_file,args.user_id,args.workspace_id)
def router(args):
 from sweep_neural_mesh.intelligence_agent import IntelligenceRouter
 return IntelligenceRouter(args.model)
def run_ask(args):
 memory=memory_adapter(args);out=execute(args.prompt,router(args),args.neural);memory.remember(f"User requested: {args.prompt}");emit(out);return 0
def run_chat(args):
 memory=memory_adapter(args);route=router(args);print("SWEEP CLI. /status /remember TEXT /recall QUERY /quit")
 while True:
  try:text=input("sweep> ").strip()
  except (EOFError,KeyboardInterrupt):print();return 0
  if text in {"/quit","/exit"}:return 0
  if text=="/status":emit(status_payload());continue
  if text.startswith("/remember "):emit({"remembered":memory.remember(text[10:])});continue
  if text.startswith("/recall "):emit({"memories":memory.recall(text[8:])});continue
  if text:emit(execute(text,route,args.neural));memory.remember(f"User requested: {text}")
def run_memory(args):
 memory=memory_adapter(args)
 if args.memory_command=="remember":emit({"remembered":memory.remember(args.text)})
 else:emit({"memories":memory.recall(args.query)})
 return 0
def run_reason(args):
 from sweep_neural_mesh.cognition import CognitiveLoop,ReasoningStore
 from sweep_neural_mesh.terminal_orchestration import query_neural
 route=router(args)
 def route_fn(text):
  value=route.classify(text);return {"domain":value["intent"],"task":value["intent"],"confidence":value.get("confidence",0.0)}
 def specialist(ctx):
  result=query_neural(ctx["request"]) if args.neural else execute(ctx["request"],route,False)
  if result.get("status")=="completed":return {"text":str(result.get("observation",""))}
  return {"text":""}
 def verifier(*unused):return {"verified":False,"confidence":0.0,"evidence_ids":[],"independent_sources":0,"uncertainty":["Production evidence verifier is not connected to this CLI path."]}
 loop=CognitiveLoop(route_fn,specialist,verifier,ReasoningStore(args.reasoning_file));conclusion=loop.run(args.prompt);emit(conclusion.to_dict());return 0
def run_explain(args):
 from sweep_neural_mesh.cognition import ReasoningStore
 value=ReasoningStore(args.reasoning_file).explain_conclusion(args.conclusion_id)
 if value is None:emit({"status":"not_found","conclusion_id":args.conclusion_id});return 2
 emit(value);return 0
def run_train(args):
 from sweep_neural_mesh.training.train_cognitive_router import train
 emit(train(args.data_dir,args.output_dir));return 0
def run_evaluate(args):
 if not args.metrics.exists():emit({"status":"unavailable","reason":"metrics file not found","path":str(args.metrics)});return 2
 emit(json.loads(args.metrics.read_text(encoding="utf-8")));return 0
def run_device(args):
 from sweep_neural_mesh.terminal_orchestration import device_plan
 if args.device_command=="plan":emit(device_plan(args.request));return 0
 emit({"status":"blocked","reason":"Use the approval-backed Device Host API for execution; CLI bypass is intentionally unavailable."});return 3
def run_capabilities(args):
 path=Path(__file__).with_name("capability_ledger.json")
 if not path.exists():emit({"status":"unavailable","reason":"capability ledger missing"});return 2
 emit(json.loads(path.read_text(encoding="utf-8")));return 0
def run_doctor(args):
 payload=status_payload();payload["checks"]={"models_present":sum(v["status"]=="available_untested" for v in payload["models"].values()),"models_blocked":sum(v["status"]=="blocked" for v in payload["models"].values()),"models_missing":sum(v["status"]=="missing" for v in payload["models"].values()),"ready_for_unverified_local_generation":any(v["status"]=="available_untested" for v in payload["models"].values())};emit(payload);return 0
def run_document(args):
 from sweep_neural_mesh.workloads import document_extract,document_inspect
 from sweep_neural_mesh.workloads_extra import document_compare
 if args.document_command=="inspect":out=document_inspect(args.path)
 elif args.document_command=="extract":out=document_extract(args.path,args.query)
 else:out=document_compare(args.left,args.right)
 emit(out);return 0 if out.get("status") in {"completed","unavailable"} else 2
def run_data(args):
 from sweep_neural_mesh.workloads import data_analyze
 from sweep_neural_mesh.workloads_extra import data_clean,data_forecast
 if args.data_command=="analyze":out=data_analyze(args.path,args.group_by)
 elif args.data_command=="clean":out=data_clean(args.path,args.output,args.fill)
 else:out=data_forecast(args.path,args.column,args.steps)
 emit(out);return 0
def run_numeric(args):
 from sweep_neural_mesh.workloads import numeric_describe,numeric_simulate
 if args.numeric_command=="describe":emit(numeric_describe(args.values))
 else:emit(numeric_simulate(args.kind,args.steps,args.seed))
 return 0
def run_research(args):
 from sweep_neural_mesh.workloads import research_fetch
 from sweep_neural_mesh.workloads_extra import research_compare
 if args.research_command=="fetch":out=research_fetch(args.url,args.max_chars)
 else:out=research_compare(args.urls,args.max_chars)
 emit(out);return 0 if out.get("status") in {"completed","blocked"} else 2
def run_export(args):
 from sweep_neural_mesh.workloads_extra import export_json
 value=json.loads(Path(args.input).read_text(encoding="utf-8"));emit(export_json(value,args.output));return 0
def add_runtime(p):
 from sweep_neural_mesh.intelligence_agent import DEFAULT_MEMORY,DEFAULT_MODEL
 p.add_argument("--model",type=Path,default=DEFAULT_MODEL);p.add_argument("--memory-file",type=Path,default=DEFAULT_MEMORY);p.add_argument("--user-id",default="terminal");p.add_argument("--workspace-id")
def parser():
 p=argparse.ArgumentParser(prog="sweep",description="SWEEP multipurpose neural-mesh CLI");p.add_argument("--version",action="version",version=VERSION);sub=p.add_subparsers(dest="command",required=True)
 ask=sub.add_parser("ask",help="route and answer a request");add_runtime(ask);ask.add_argument("--neural",action="store_true");ask.add_argument("prompt");ask.set_defaults(func=run_ask)
 chat=sub.add_parser("chat",help="interactive terminal assistant");add_runtime(chat);chat.add_argument("--neural",action="store_true");chat.set_defaults(func=run_chat)
 st=sub.add_parser("status",help="show runtime status");st.set_defaults(func=lambda a:(emit(status_payload()) or 0))
 models=sub.add_parser("models",help="probe governed local specialists");models.set_defaults(func=lambda a:(emit(model_status()) or 0))
 mem=sub.add_parser("memory",help="persistent memory");add_runtime(mem);m=mem.add_subparsers(dest="memory_command",required=True);r=m.add_parser("remember");r.add_argument("text");r.set_defaults(func=run_memory);q=m.add_parser("recall");q.add_argument("query");q.set_defaults(func=run_memory)
 reason=sub.add_parser("reason",help="run bounded cognitive loop");add_runtime(reason);reason.add_argument("--neural",action="store_true");reason.add_argument("--reasoning-file",type=Path,default=DEFAULT_REASONING);reason.add_argument("prompt");reason.set_defaults(func=run_reason)
 explain=sub.add_parser("explain",help="load a persisted conclusion audit");explain.add_argument("--reasoning-file",type=Path,default=DEFAULT_REASONING);explain.add_argument("conclusion_id");explain.set_defaults(func=run_explain)
 trainp=sub.add_parser("train",help="run bounded training");trainp.add_argument("training_target",choices=["cognitive-router"]);trainp.add_argument("--data-dir",type=Path,default=Path("sweep_neural_mesh/training/datasets"));trainp.add_argument("--output-dir",type=Path,default=DEFAULT_COGNITIVE_ARTIFACT);trainp.set_defaults(func=run_train)
 ev=sub.add_parser("evaluate",help="display persisted held-out metrics");ev.add_argument("evaluation_target",choices=["cognitive-router"]);ev.add_argument("--metrics",type=Path,default=Path("docs/metrics/cognitive_router_metrics.json"));ev.set_defaults(func=run_evaluate)
 dev=sub.add_parser("device",help="permission-gated device operations");d=dev.add_subparsers(dest="device_command",required=True);planp=d.add_parser("plan");planp.add_argument("request");planp.set_defaults(func=run_device);runp=d.add_parser("run");runp.set_defaults(func=run_device)
 caps=sub.add_parser("capabilities",help="show verified/candidate/blocked capabilities");caps.set_defaults(func=run_capabilities)
 doctor=sub.add_parser("doctor",help="diagnose local readiness");doctor.set_defaults(func=run_doctor)
 document=sub.add_parser("document",help="inspect extract or compare local documents");doc=document.add_subparsers(dest="document_command",required=True);inspect=doc.add_parser("inspect");inspect.add_argument("path");inspect.set_defaults(func=run_document);extract=doc.add_parser("extract");extract.add_argument("path");extract.add_argument("--query");extract.set_defaults(func=run_document);compare=doc.add_parser("compare");compare.add_argument("left");compare.add_argument("right");compare.set_defaults(func=run_document)
 data=sub.add_parser("data",help="analyze clean or forecast local tabular data");data.add_subparsers(dest="data_command",required=True);analyze=data._subparsers._group_actions[0].add_parser("analyze");analyze.add_argument("path");analyze.add_argument("--group-by");analyze.set_defaults(func=run_data);clean=data._subparsers._group_actions[0].add_parser("clean");clean.add_argument("path");clean.add_argument("--output");clean.add_argument("--fill",choices=["empty","zero","mean"],default="empty");clean.set_defaults(func=run_data);forecast=data._subparsers._group_actions[0].add_parser("forecast");forecast.add_argument("path");forecast.add_argument("column");forecast.add_argument("--steps",type=int,default=5);forecast.set_defaults(func=run_data)
 numeric=sub.add_parser("numeric",help="numeric summaries and simulations");num=numeric.add_subparsers(dest="numeric_command",required=True);describe=num.add_parser("describe");describe.add_argument("values",nargs="+",type=float);describe.set_defaults(func=run_numeric);simulate=num.add_parser("simulate");simulate.add_argument("kind",choices=["random-walk","logistic-growth"]);simulate.add_argument("--steps",type=int,default=100);simulate.add_argument("--seed",type=int,default=0);simulate.set_defaults(func=run_numeric)
 research=sub.add_parser("research",help="fetch public URLs with safety guards");res=research.add_subparsers(dest="research_command",required=True);fetch=res.add_parser("fetch");fetch.add_argument("url");fetch.add_argument("--max-chars",type=int,default=60000);fetch.set_defaults(func=run_research);compare=res.add_parser("compare");compare.add_argument("urls",nargs="+");compare.add_argument("--max-chars",type=int,default=12000);compare.set_defaults(func=run_research)
 export=sub.add_parser("export",help="export JSON results");ex=export.add_subparsers(dest="export_command",required=True);j=ex.add_parser("json");j.add_argument("input");j.add_argument("output");j.set_defaults(func=run_export)
 return p
def legacy_main(argv):
 p=argparse.ArgumentParser();add_runtime(p);p.add_argument("--neural",action="store_true");p.add_argument("--once");p.add_argument("--remember");p.add_argument("--recall");p.add_argument("--status",action="store_true");a=p.parse_args(argv)
 if a.status:emit(status_payload());return 0
 if a.remember is not None:return run_memory(argparse.Namespace(**vars(a),memory_command="remember",text=a.remember))
 if a.recall is not None:return run_memory(argparse.Namespace(**vars(a),memory_command="recall",query=a.recall))
 if a.once is not None:return run_ask(argparse.Namespace(**vars(a),prompt=a.once))
 return run_chat(a)
def main(argv=None):
 argv=list(sys.argv[1:] if argv is None else argv)
 try:
  if not argv or any(x in argv for x in ("--once","--remember","--recall","--status")):return legacy_main(argv)
  args=parser().parse_args(argv);return int(args.func(args) or 0)
 except BrokenPipeError:return 0
 except Exception as exc:emit({"status":"error","error_type":type(exc).__name__,"message":str(exc)});return 1
if __name__=="__main__":raise SystemExit(main())
