"""Bounded chart generation through the existing companion data tool."""
from __future__ import annotations
import base64
from pathlib import Path
from typing import Any
from .workloads import _file,_rows,_sha

def data_plot(path:str,kind:str="line",x:str|None=None,y:str|None=None,output:str|None=None)->dict[str,Any]:
 try:p=_file(path)
 except ValueError as exc:return {"status":"error","path":str(path),"reason":str(exc)}
 if not output:return {"status":"error","path":str(p),"sha256":_sha(p),"reason":"An output path is required for chart generation."}
 out=Path(output).expanduser().resolve()
 if out==p:return {"status":"error","path":str(p),"sha256":_sha(p),"reason":"Output path must differ from input path."}
 try:
  columns,rows=_rows(p)
  if not rows:return {"status":"error","path":str(p),"sha256":_sha(p),"reason":"The input dataset has no rows."}
  from companion.tools.data import run_plot
  params={"kind":kind}
  if x:params["x"]=x
  if y:params["y"]=y
  result=run_plot({"data":rows,"params":params})
  png=(result.get("result") or {}).get("png_base64")
  if not png:raise RuntimeError("Plot adapter returned no PNG output.")
  out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(base64.b64decode(png))
  return {"status":"completed","input":{"path":str(p),"sha256":_sha(p),"columns":columns,"rows":len(rows)},"output":{"path":str(out),"sha256":_sha(out),"bytes":out.stat().st_size},"kind":kind,"summary":result.get("summary"),"libraries_used":result.get("libraries_used",[])}
 except Exception as exc:
  return {"status":"unavailable","path":str(p),"sha256":_sha(p),"reason":f"Chart generation unavailable: {type(exc).__name__}: {str(exc)[:200]}"}
