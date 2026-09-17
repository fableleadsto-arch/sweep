"""Train SWEEP cognitive domain/task routers on immutable repository splits."""
from __future__ import annotations
import argparse,hashlib,json,pickle,platform,random,sys,time
from pathlib import Path
SEED=20260917
def sha256(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load_rows(path,expected_split):
 rows=[]
 for line_no,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
  if not line.strip():continue
  row=json.loads(line)
  if row.get("split")!=expected_split:raise ValueError(f"{path}:{line_no} split mismatch")
  if not row.get("input_text") or not row.get("domain") or not row.get("task"):raise ValueError(f"{path}:{line_no} required field missing")
  rows.append(row)
 return rows
def text(row):return row["input_text"]+"\n"+"\n".join(str(x) for x in row.get("evidence",[]))
def tokens(value):return set(value.lower().split())
def contamination(train,evaluation,threshold=.9):
 train_text={text(r).strip().lower() for r in train};eval_text={text(r).strip().lower() for r in evaluation};near=0
 for a in train_text:
  ta=tokens(a)
  for b in eval_text:
   tb=tokens(b);union=ta|tb
   if a!=b and union and len(ta&tb)/len(union)>=threshold:near+=1
 return {"exact":len(train_text&eval_text),"near_jaccard_0_9_pairs":near}
def bootstrap_accuracy(truth,pred,resamples=2000):
 rng=random.Random(SEED);n=len(truth);scores=[]
 for _ in range(resamples):
  idx=[rng.randrange(n) for _ in range(n)];scores.append(sum(truth[i]==pred[i] for i in idx)/n)
 scores.sort();return [scores[int(.025*resamples)],scores[min(resamples-1,int(.975*resamples))]]
def train(data_dir:Path,out:Path):
 import sklearn
 from sklearn.feature_extraction.text import TfidfVectorizer
 from sklearn.metrics import accuracy_score,f1_score
 from sklearn.pipeline import Pipeline
 from sklearn.svm import LinearSVC
 paths={s:data_dir/f"comprehensive_{s}.jsonl" for s in ("train","val","test")};rows={s:load_rows(p,s) for s,p in paths.items()};started=time.perf_counter();models={};metrics={}
 for head in ("domain","task"):
  model=Pipeline([("tfidf",TfidfVectorizer(lowercase=True,ngram_range=(1,2),sublinear_tf=True,max_features=30000)),("classifier",LinearSVC(class_weight="balanced",random_state=SEED))]);model.fit([text(r) for r in rows["train"]],[r[head] for r in rows["train"]]);models[head]=model
  def evaluate(split):
   truth=[r[head] for r in rows[split]];pred=model.predict([text(r) for r in rows[split]]).tolist();return {"examples":len(truth),"accuracy":accuracy_score(truth,pred),"macro_f1":f1_score(truth,pred,average="macro",zero_division=0),"accuracy_95pct_bootstrap_ci":bootstrap_accuracy(truth,pred),"labels":sorted(set(truth))}
  metrics[head]={"validation":evaluate("val"),"test":evaluate("test")}
 report={"version":"sweep-cognitive-router/1","seed":SEED,"purpose":"routing only; not proof of reasoning quality","files":{s:{"sha256":sha256(p),"rows":len(rows[s])} for s,p in paths.items()},"contamination":{"train_validation":contamination(rows["train"],rows["val"]),"train_test":contamination(rows["train"],rows["test"]),"validation_test":contamination(rows["val"],rows["test"])},"heads":metrics,"training_seconds":time.perf_counter()-started,"environment":{"python":sys.version,"platform":platform.platform(),"scikit_learn":sklearn.__version__}}
 out.mkdir(parents=True,exist_ok=True)
 with (out/"cognitive_router.pkl").open("wb") as f:pickle.dump({"version":report["version"],"models":models},f)
 (out/"cognitive_router_metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8");return report
def main():
 p=argparse.ArgumentParser();p.add_argument("--data-dir",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args();print(json.dumps(train(a.data_dir,a.output_dir),indent=2))
if __name__=="__main__":main()
