"""Bounded report generation from persisted SWEEP JSON results."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .workloads import _file,_sha

def report_markdown(input_path:str,output_path:str)->dict[str,Any]:
 try:source=_file(input_path)
 except ValueError as exc:return {"status":"error","path":str(input_path),"reason":str(exc)}
 output=Path(output_path).expanduser().resolve()
 if output==source:return {"status":"error","path":str(source),"reason":"Output path must differ from input path."}
 try:value=json.loads(source.read_text(encoding="utf-8"))
 except (OSError,json.JSONDecodeError) as exc:return {"status":"error","path":str(source),"sha256":_sha(source),"reason":f"JSON report input is invalid: {exc}"}
 status=str(value.get("status", "unreported")) if isinstance(value,dict) else "unreported"
 summary=value.get("summary", "") if isinstance(value,dict) else ""
 body=["# SWEEP Report","",f"- **Input:** `{source}`",f"- **Input SHA-256:** `{_sha(source)}`",f"- **Status:** `{status}`"]
 if summary:body += ["", "## Summary", "", str(summary)]
 body += ["", "## Result", "", "```json", json.dumps(value,indent=2,ensure_ascii=False,default=str), "```", "", "_This report is a rendering of supplied JSON. It does not independently verify the underlying claims._", ""]
 output.parent.mkdir(parents=True,exist_ok=True);output.write_text("\n".join(body),encoding="utf-8")
 return {"status":"completed","input":{"path":str(source),"sha256":_sha(source)},"output":{"path":str(output),"sha256":_sha(output),"bytes":output.stat().st_size},"claims_verified":False}
