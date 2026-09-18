"""Validate and normalize a pinned CLINC150 file without changing its splits."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
EXPECTED_BLOB="7a7b26c5f2dfbbf213f3e67d2dd0727e1af545aa"
SOURCE="https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/data/data_full.json"
def git_blob_sha1(data:bytes)->str:return hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()
def normalize(path:Path,out_dir:Path,allow_unpinned:bool=False):
 data_bytes=path.read_bytes();actual=git_blob_sha1(data_bytes)
 if actual!=EXPECTED_BLOB and not allow_unpinned:raise ValueError(f"CLINC150 blob mismatch: {actual}")
 raw=json.loads(data_bytes);out_dir.mkdir(parents=True,exist_ok=True);counts={}
 for split in ("train","val","test","oos_train","oos_val","oos_test"):
  rows=raw.get(split,[]);counts[split]=len(rows)
  with (out_dir/f"clinc150_{split}.jsonl").open("w",encoding="utf-8") as f:
   for i,item in enumerate(rows):
    text,label=(item if isinstance(item,list) else (item.get("text",""),item.get("label","oos")))
    f.write(json.dumps({"id":f"CLINC150-{split}-{i:06d}","input":text,"expected_output":label,"split":split,"source":SOURCE,"license":"CC-BY-3.0"})+"\n")
 report={"source":SOURCE,"git_blob_sha1":actual,"pinned":actual==EXPECTED_BLOB,"counts":counts,"license":"CC-BY-3.0","attribution_required":True,"splits_preserved":True};(out_dir/"clinc150_import_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8");return report
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--allow-unpinned-fixture",action="store_true");a=p.parse_args();print(json.dumps(normalize(a.input,a.output_dir,a.allow_unpinned_fixture),indent=2))
if __name__=="__main__":main()
