"""Train/evaluate CLINC150 with immutable splits and validation-only OOS calibration."""
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
 import numpy as np,sklearn
 from sklearn.feature_extraction.text import TfidfVectorizer
 from sklearn.metrics import accuracy_score,f1_score
 from sklearn.pipeline import Pipeline
 from sklearn.svm import LinearSVC
 data=source.read_bytes();actual=blob_sha(data)
 if actual!=EXPECTED_BLOB and not allow_unpinned:raise ValueError(f"CLINC150 blob mismatch: {actual}")
 raw=json.loads(data);raw_train=pairs(raw,"train");val_rows=pairs(raw,"val","oos_val");test_rows=pairs(raw,"test","oos_test")
 if not raw_train or not val_rows or not test_rows:raise ValueError("required CLINC150 splits missing")
 validation_text={x for x,_ in val_rows};test_text={x for x,_ in test_rows};removed=[r for r in raw_train if r[0] in validation_text or r[0] in test_text];train_rows=[r for r in raw_train if r not in removed]
 model=Pipeline([("tfidf",TfidfVectorizer(lowercase=True,ngram_range=(1,2),sublinear_tf=True,min_df=1,max_features=75000)),("classifier",LinearSVC(C=1.0,class_weight="balanced",random_state=SEED))])
 t0=time.perf_counter();model.fit([x for x,_ in train_rows],[y for _,y in train_rows]);elapsed=time.perf_counter()-t0
 classes=model.named_steps["classifier"].classes_
 def scored(rows):
  truth=[y for _,y in rows];decision=model.decision_function([x for x,_ in rows]);indices=np.argmax(decision,axis=1);return truth,[str(classes[i]) for i in indices],np.max(decision,axis=1)
 val_truth,val_base,val_scores=scored(val_rows)
 ordered=np.sort(np.unique(val_scores));candidate_indices=np.linspace(0,len(ordered)-1,num=min(501,len(ordered)),dtype=int);candidates=[float(ordered[i]) for i in candidate_indices]
 def apply(base,scores,threshold):return ["oos" if float(score)<threshold else pred for pred,score in zip(base,scores)]
 def balanced(truth,pred):
  ins=[i for i,y in enumerate(truth) if y!="oos"];oos=[i for i,y in enumerate(truth) if y=="oos"];in_acc=sum(pred[i]==truth[i] for i in ins)/len(ins);oos_rec=sum(pred[i]=="oos" for i in oos)/len(oos);return (in_acc+oos_rec)/2
 ranked=[(balanced(val_truth,apply(val_base,val_scores,t)),t) for t in candidates];validation_balanced,threshold=max(ranked,key=lambda x:(x[0],-x[1]))
 def evaluate(rows):
  truth,base,scores=scored(rows);pred=apply(base,scores,threshold);ins=[i for i,y in enumerate(truth) if y!="oos"];oos=[i for i,y in enumerate(truth) if y=="oos"]
  return {"examples":len(rows),"accuracy":accuracy_score(truth,pred),"macro_f1":f1_score(truth,pred,average="macro",zero_division=0),"in_scope_accuracy":accuracy_score([truth[i] for i in ins],[pred[i] for i in ins]),"oos_recall":sum(pred[i]=="oos" for i in oos)/len(oos),"balanced_in_scope_oos":balanced(truth,pred)}
 train_text={x for x,_ in train_rows};majority=Counter(y for _,y in train_rows).most_common(1)[0][1]/len(train_rows)
 report={"version":"sweep-clinc150-training/2","source":{"git_blob_sha1":actual,"expected":EXPECTED_BLOB,"pinned":actual==EXPECTED_BLOB,"license":"CC-BY-3.0","attribution_required":True},"seed":SEED,"model":"word_1_2gram_tfidf_linear_svc_validation_oos_threshold","counts":{"raw_train":len(raw_train),"train_after_decontamination":len(train_rows),"validation":len(val_rows),"test":len(test_rows)},"contamination_audit":{"removed_from_train":len(removed),"removed_text_sha256":[hashlib.sha256(x.encode()).hexdigest() for x,_ in removed],"post_train_validation_overlap":len(train_text&validation_text),"post_train_test_overlap":len(train_text&test_text),"validation_test_overlap":len(validation_text&test_text)},"calibration":{"source":"validation_only","objective":"mean(in_scope_accuracy,oos_recall)","threshold":threshold,"validation_objective":validation_balanced,"candidate_count":len(candidates)},"baseline_train_majority_fraction":majority,"validation":evaluate(val_rows),"test":evaluate(test_rows),"training_seconds":elapsed,"environment":{"python":sys.version,"platform":platform.platform(),"scikit_learn":sklearn.__version__}}
 out.mkdir(parents=True,exist_ok=True)
 with (out/"clinc150_intent_router.pkl").open("wb") as f:pickle.dump({"model":model,"oos_threshold":threshold,"version":report["version"]},f)
 (out/"clinc150_metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8");return report
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--allow-unpinned-fixture",action="store_true");a=p.parse_args();print(json.dumps(train(a.input,a.output_dir,a.allow_unpinned_fixture),indent=2))
if __name__=="__main__":main()
