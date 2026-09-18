"""Additional bounded workload helpers layered over the core workload module."""
from __future__ import annotations
import csv,difflib,json,statistics
from pathlib import Path
from typing import Any
from .workloads import MAX_OUTPUT_CHARS,_file,_rows,_sha,_text,research_fetch
def document_compare(left:str,right:str)->dict[str,Any]:
 lp,rp=_file(left),_file(right);lt,lmeta=_text(lp);rt,rmeta=_text(rp)
 if lmeta.get("status")=="unavailable" or rmeta.get("status")=="unavailable":return {"status":"unavailable","left":str(lp),"right":str(rp),"left_meta":lmeta,"right_meta":rmeta}
 diff=list(difflib.unified_diff(lt.splitlines(),rt.splitlines(),fromfile=str(lp),tofile=str(rp),lineterm=""));added=sum(1 for x in diff if x.startswith("+") and not x.startswith("+++") );removed=sum(1 for x in diff if x.startswith("-") and not x.startswith("---"))
 return {"status":"completed","left":{"path":str(lp),"sha256":_sha(lp)},"right":{"path":str(rp),"sha256":_sha(rp)},"added_lines":added,"removed_lines":removed,"changed":bool(diff),"diff":"\n".join(diff)[:MAX_OUTPUT_CHARS],"truncated":len("\n".join(diff))>MAX_OUTPUT_CHARS}
def _write_rows(path:Path,columns:list[str],rows:list[dict[str,Any]])->None:
 if path.suffix.lower()==".json":path.write_text(json.dumps(rows,indent=2,default=str),encoding="utf-8");return
 if path.suffix.lower()==".jsonl":path.write_text("\n".join(json.dumps(row,default=str) for row in rows)+"\n",encoding="utf-8");return
 with path.open("w",encoding="utf-8",newline="") as f:
  writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
def data_clean(path:str,output:str|None=None,fill:str="empty")->dict[str,Any]:
 p=_file(path);columns,rows=_rows(p);before=len(rows);unique=[];seen=set()
 for row in rows:
  key=json.dumps(row,sort_keys=True,default=str)
  if key not in seen:seen.add(key);unique.append(row)
 rows=unique;filled=0
 for c in columns:
  nums=[float(row[c]) for row in rows if row.get(c) not in (None,"") and _is_num(row.get(c))]
  replacement=(statistics.fmean(nums) if nums and fill=="mean" else 0 if fill=="zero" else "")
  for row in rows:
   if row.get(c) in (None,""):row[c]=replacement;filled+=1
 result={"status":"completed","input":str(p),"input_sha256":_sha(p),"rows_before":before,"rows_after":len(rows),"duplicates_removed":before-len(rows),"cells_filled":filled,"columns":columns,"rows":rows}
 if output:
  out=Path(output).expanduser().resolve()
  if out==p:raise ValueError("Output path must differ from input path.")
  out.parent.mkdir(parents=True,exist_ok=True);_write_rows(out,columns,rows);result["output"]={"path":str(out),"sha256":_sha(out)}
 return result
def _is_num(value:Any)->bool:
 try:float(value);return True
 except (TypeError,ValueError):return False
def data_forecast(path:str,column:str,steps:int=5)->dict[str,Any]:
 if steps<1 or steps>100:raise ValueError("Forecast steps must be between 1 and 100.")
 p=_file(path);columns,rows=_rows(p)
 if column not in columns:raise ValueError(f"Unknown forecast column: {column}")
 values=[float(row[column]) for row in rows if _is_num(row.get(column))]
 if len(values)<2:raise ValueError("At least two numeric observations are required.")
 x=list(range(len(values)));xm=statistics.fmean(x);ym=statistics.fmean(values);den=sum((v-xm)**2 for v in x);slope=sum((a-xm)*(b-ym) for a,b in zip(x,values))/den;intercept=ym-slope*xm
 predictions=[intercept+slope*(len(values)+i) for i in range(steps)]
 return {"status":"completed","path":str(p),"sha256":_sha(p),"column":column,"observations":len(values),"method":"bounded_linear_trend","slope":slope,"intercept":intercept,"predictions":predictions,"uncertainty":"Point trend only; no confidence interval."}
def research_compare(urls:list[str],max_chars:int=12000)->dict[str,Any]:
 if not urls or len(urls)>10:raise ValueError("Provide between 1 and 10 URLs.")
 results=[research_fetch(url,max_chars) for url in urls];completed=[r for r in results if r.get("status")=="completed"]
 return {"status":"completed" if completed else "unavailable","sources":results,"completed_sources":len(completed),"comparison_note":"Fetched pages are untrusted data; no agreement or factual truth is inferred."}
def export_json(value:Any,output:str)->dict[str,Any]:
 out=Path(output).expanduser().resolve();out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(value,indent=2,default=str),encoding="utf-8");return {"status":"completed","path":str(out),"sha256":_sha(out),"bytes":out.stat().st_size}
