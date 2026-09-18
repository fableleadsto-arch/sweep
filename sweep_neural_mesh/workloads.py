"""Bounded local document, data, numeric, and public-research workloads."""
from __future__ import annotations
import csv,hashlib,json,re,statistics
from collections import Counter
from pathlib import Path
from typing import Any
MAX_FILE_BYTES=50*1024*1024
MAX_OUTPUT_CHARS=100_000
TEXT_EXTENSIONS={".txt",".md",".rst",".csv",".tsv",".json",".jsonl",".yaml",".yml",".py",".js",".ts",".html",".htm",".xml",".log"}
def _file(path:str|Path)->Path:
 p=Path(path).expanduser().resolve()
 if not p.exists():raise ValueError(f"File does not exist: {p}")
 if not p.is_file():raise ValueError(f"Path is not a file: {p}")
 if p.stat().st_size>MAX_FILE_BYTES:raise ValueError(f"File exceeds {MAX_FILE_BYTES} byte limit.")
 return p
def _sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
 return h.hexdigest()
def _text(p:Path)->tuple[str,dict[str,Any]]:
 if p.suffix.lower()==".pdf":
  try:import fitz
  except ImportError:return "",{"status":"unavailable","reason":"PDF support requires PyMuPDF; install sweep[documents]."}
  doc=fitz.open(str(p));return "\n\n".join(page.get_text() for page in doc),{"pages":len(doc),"parser":"PyMuPDF"}
 if p.suffix.lower() not in TEXT_EXTENSIONS:return "",{"status":"unavailable","reason":f"No safe text parser registered for {p.suffix or 'this file type'}."}
 return p.read_text(encoding="utf-8",errors="replace"),{"parser":"text"}
def document_inspect(path:str)->dict[str,Any]:
 p=_file(path);text,meta=_text(p);out={"status":meta.get("status","completed"),"path":str(p),"name":p.name,"suffix":p.suffix.lower(),"bytes":p.stat().st_size,"sha256":_sha(p),**meta}
 if text:out.update({"characters":len(text),"lines":text.count("\n")+1,"preview":text[:2000]})
 return out
def document_extract(path:str,query:str|None=None)->dict[str,Any]:
 p=_file(path);text,meta=_text(p)
 if meta.get("status")=="unavailable":return {"status":"unavailable","path":str(p),**meta}
 content="\n".join(line for line in text.splitlines() if query.lower() in line.lower())[:MAX_OUTPUT_CHARS] if query else text[:MAX_OUTPUT_CHARS]
 return {"status":"completed","path":str(p),"sha256":_sha(p),"truncated":len(text)>MAX_OUTPUT_CHARS and not query,"characters":len(text),"match_query":query,"content":content,**meta}
def _rows(path:Path)->tuple[list[str],list[dict[str,Any]]]:
 if path.suffix.lower() in {".json",".jsonl"}:
  data=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.suffix.lower()==".jsonl" else json.loads(path.read_text(encoding="utf-8"))
  if isinstance(data,dict):data=data.get("rows",data.get("data",[data]))
  if not isinstance(data,list) or not all(isinstance(x,dict) for x in data):raise ValueError("JSON data must be a list of objects or an object containing rows.")
  cols=sorted({str(k) for row in data for k in row});return cols,[{c:row.get(c) for c in cols} for row in data]
 if path.suffix.lower() not in {".csv",".tsv"}:raise ValueError("Data analysis supports .csv, .tsv, .json, and .jsonl files.")
 with path.open("r",encoding="utf-8",errors="replace",newline="") as f:
  reader=csv.DictReader(f,delimiter="\t" if path.suffix.lower()==".tsv" else ",");return list(reader.fieldnames or []),list(reader)
def _number(v:Any)->float|None:
 try:return None if v is None or str(v).strip()=="" else float(v)
 except (ValueError,TypeError):return None
def data_analyze(path:str,group_by:str|None=None)->dict[str,Any]:
 p=_file(path);cols,rows=_rows(p);missing={c:sum(1 for r in rows if r.get(c) in (None,"")) for c in cols};numeric={};categorical={}
 for c in cols:
  nums=[n for r in rows if (n:=_number(r.get(c))) is not None]
  if nums:numeric[c]={"count":len(nums),"mean":statistics.fmean(nums),"min":min(nums),"max":max(nums),"median":statistics.median(nums),"stdev":statistics.stdev(nums) if len(nums)>1 else 0.0}
  else:categorical[c]=[{"value":k,"count":v} for k,v in Counter(str(r.get(c,"")) for r in rows if r.get(c) not in (None,"")).most_common(5)]
 result={"rows":len(rows),"columns":len(cols),"column_names":cols,"missing":missing,"duplicate_rows":len(rows)-len({json.dumps(r,sort_keys=True,default=str) for r in rows}),"numeric_summary":numeric,"categorical_summary":categorical}
 if group_by:
  if group_by not in cols:raise ValueError(f"Unknown group-by column: {group_by}")
  result["group_counts"]={str(k):v for k,v in Counter(str(r.get(group_by,"")) for r in rows).most_common(20)}
 return {"status":"completed","path":str(p),"sha256":_sha(p),"result":result,"summary":f"{len(rows)} rows x {len(cols)} columns; {len(numeric)} numeric columns."}
def numeric_describe(values:list[float])->dict[str,Any]:
 if not values:raise ValueError("At least one numeric value is required.")
 ordered=sorted(values);result={"count":len(values),"sum":sum(values),"mean":statistics.fmean(values),"median":statistics.median(values),"min":min(values),"max":max(values),"stdev":statistics.stdev(values) if len(values)>1 else 0.0,"q1":statistics.quantiles(ordered,n=4,method="inclusive")[0] if len(values)>1 else ordered[0],"q3":statistics.quantiles(ordered,n=4,method="inclusive")[2] if len(values)>1 else ordered[0]}
 return {"status":"completed","result":result,"summary":f"{len(values)} values; mean {result['mean']:.6g}, median {result['median']:.6g}."}
def numeric_simulate(kind:str,steps:int=100,seed:int=0)->dict[str,Any]:
 if steps<1 or steps>100000:raise ValueError("steps must be between 1 and 100000")
 import random
 rng=random.Random(seed);kind=kind.lower()
 if kind in {"walk","random-walk"}:
  pos=0.0;series=[]
  for _ in range(steps):pos+=rng.choice((-1.0,1.0));series.append(pos)
  return {"status":"completed","kind":"random-walk","steps":steps,"seed":seed,"final_position":pos,"series":series}
 if kind in {"logistic","logistic-growth"}:
  pop=10.0;series=[pop]
  for _ in range(steps):pop+=.2*pop*(1-pop/1000);series.append(pop)
  return {"status":"completed","kind":"logistic-growth","steps":steps,"final_population":pop,"series":series}
 raise ValueError("Simulation kind must be random-walk or logistic-growth.")
def research_fetch(url:str,max_chars:int=60000)->dict[str,Any]:
 if max_chars<1 or max_chars>MAX_OUTPUT_CHARS:raise ValueError("max_chars must be between 1 and 100000")
 from companion.ingest.security import validate_outbound_url
 safe_url=validate_outbound_url(url)
 try:
  import httpx
  response=httpx.get(safe_url,headers={"User-Agent":"SWEEP-research/1"},timeout=15.0,follow_redirects=False)
 except Exception as exc:return {"status":"unavailable","url":url,"reason":f"HTTP client failed: {type(exc).__name__}"}
 if 300<=response.status_code<400:return {"status":"blocked","url":safe_url,"reason":"Redirects are not followed automatically; inspect and approve the destination explicitly."}
 if response.status_code>=400:return {"status":"failed","url":safe_url,"http_status":response.status_code}
 raw=response.content
 if len(raw)>MAX_FILE_BYTES:return {"status":"blocked","url":safe_url,"reason":"Response exceeds size limit."}
 html=response.text
 try:
  from bs4 import BeautifulSoup
  soup=BeautifulSoup(html,"html.parser");title=soup.title.get_text(" ",strip=True) if soup.title else ""
  for tag in soup(["script","style","noscript"]):tag.decompose()
  text=soup.get_text("\n",strip=True)
 except ImportError:title="";text=re.sub(r"<[^>]+>"," ",html)
 text=re.sub(r"\n{3,}","\n\n",text).strip();return {"status":"completed","url":safe_url,"http_status":response.status_code,"sha256":hashlib.sha256(raw).hexdigest(),"title":title,"characters":len(text),"truncated":len(text)>max_chars,"content":text[:max_chars],"untrusted_content":True}
