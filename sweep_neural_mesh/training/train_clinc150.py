"""Train/evaluate a CPU CLINC150 baseline using immutable upstream splits."""
from __future__ import annotations
import argparse,hashlib,json,pickle,platform,sys,time
from collections import Counter
from pathlib import Path
EXPECTED_BLOB="7a7b26c5f2dfbbf213f3e67d2dd0727e1af545aa";SEED=20260917
def blob_sha(data):return hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()
def pairs(raw,*keys):
 out=[]
 for key in keys:
  for item in raw.get(key,[]):out.append((str(item[0]),str(item[1])))
 return out
def train(source:Path,out:Path,allow_unpinned=False):
 import sklearn
 from sklearn.feature_extraction.text import TfidfVectorizer
 from sklearn.metrics import accuracy_score,f1_score
 from sklearn.pipeline import Pipeline
 from sklearn.svm import LinearSVC
 data=source.read_bytes();actual=blob_sha(data)
 if actual!=EXPECTED_BLOB and not allow_unpinned:raise ValueError(f"CLINC150 blob mismatch: {actual}")
 raw=json.loads(data);train_rows=pairs(raw,"train","oos_train");val_rows=pairs(raw,"val","oos_val");test_rows=pairs(raw,"test","oos_test")
 if not train_rows or not val_rows or not test_rows:raise ValueError("required CLINC150 splits missing")
 model=Pipeline([("tfidf",TfidfVectorizer(lowercase=True,ngram_range=(1,2),sublinear_tf=True,min_df=1,max_features=75000)),("classifier",LinearSVC(C=1.0,class_weight="balanced",random_state=SEED))])
 t0=time.perf_counter();model.fit([x for x,_ in train_rows],[y for _,y in train_rows]);elapsed=time.perf_counter()-t0
 def evaluate(rows):
  truth=[y for _,y in rows];pred=model.predict([x for x,_ in rows]).tolist();ins=[i for i,y in enumerate(truth) if y!="oos"];oos=[i for i,y in enumerate(truth) if y=="oos"]
  return {"examples":len(rows),"accuracy":accuracy_score(truth,pred),"macro_f1":f1_score(truth,pred,average="macro",zero_division=0),"in_scope_accuracy":accuracy_score([truth[i] for i in ins],[pred[i] for i in ins]) if ins else None,"oos_recall":sum(pred[i]=="oos" for i in oos)/len(oos) if oos else None}
 train_text={x for x,_ in train_rows};val_text={x for x,_ in val_rows};test_text={x for x,_ in test_rows};majority=Counter(y for _,y in train_rows).most_common(1)[0][1]/len(train_rows)
 report={"version":"sweep-clinc150-training/1","source":{"git_blob_sha1":actual,"expected":EXPECTED_BLOB,"pinned":actual==EXPECTED_BLOB,"license":"CC-BY-3.0","attribution_required":True},"seed":SEED,"model":"word_1_2gram_tfidf_linear_svc","counts":{"train":len(train_rows),"validation":len(val_rows),"test":len(test_rows)},"contamination_audit":{"train_validation_exact_text_overlap":len(train_text&val_text),"train_test_exact_text_overlap":len(train_text&test_text),"validation_test_exact_text_overlap":len(val_text&test_text)},"baseline_train_majority_fraction":majority,"validation":evaluate(val_rows),"test":evaluate(test_rows),"training_seconds":elapsed,"environment":{"python":sys.version,"platform":platform.platform(),"scikit_learn":sklearn.__version__}}
 out.mkdir(parents=True,exist_ok=True)
 with (out/"clinc150_intent_router.pkl").open("wb") as f:pickle.dump(model,f)
 (out/"clinc150_metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8");return report
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--allow-unpinned-fixture",action="store_true");a=p.parse_args();print(json.dumps(train(a.input,a.output_dir,a.allow_unpinned_fixture),indent=2))
if __name__=="__main__":main()
