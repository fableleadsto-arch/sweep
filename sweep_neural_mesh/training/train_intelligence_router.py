"""Reproducible CPU training for SWEEP's intelligence-task router."""
from __future__ import annotations
import argparse,hashlib,json,math,random,re
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
VERSION="sweep-intelligence-router/1";SEED=20260917
LABELS=("investigation","search","evidence_analysis","comparison","timeline","relationship_analysis","location_analysis","source_verification","contradiction_analysis","summarization","extraction","device_action","unknown_ambiguous")
TRAIN={
"investigation":["investigate {x} using public records","build an intelligence brief about {x}","research the background of {x}","perform lawful due diligence regarding {x}","develop a public-source dossier concerning {x}"],
"search":["search for current information about {x}","find public sources about {x}","look up reports concerning {x}","locate open web information about {x}","retrieve recent articles concerning {x}"],
"evidence_analysis":["assess the evidence supporting {x}","how strong is the proof for {x}","evaluate evidence about {x}","weigh corroborating material for {x}","grade the factual support behind {x}"],
"comparison":["compare {x} with {y}","what differs between {x} and {y}","contrast {x} versus {y}","analyze similarities between {x} and {y}","which is better supported {x} or {y}"],
"timeline":["create a timeline for {x}","put the events about {x} in chronological order","what happened before and after {x}","reconstruct the event sequence for {x}","order all dated events linked to {x}"],
"relationship_analysis":["map relationships involving {x}","how is {x} connected to {y}","find affiliations between {x} and {y}","construct an association network around {x}","trace organizational links between {x} and {y}"],
"location_analysis":["analyze public locations associated with {x}","where is {x} publicly located","map places connected to {x}","geolocate an authorized event concerning {x}","list known public places tied to {x}"],
"source_verification":["verify whether the source about {x} is credible","check the reliability of this report on {x}","is this source trustworthy about {x}","audit provenance and credibility concerning {x}","rate the trustworthiness of a citation about {x}"],
"contradiction_analysis":["find contradictions about {x}","do these claims concerning {x} conflict","check whether reports on {x} disagree","detect inconsistent statements about {x}","identify mutually exclusive claims involving {x}"],
"summarization":["summarize the report about {x}","give the key points concerning {x}","produce a concise overview of {x}","condense the findings concerning {x}","brief me on conclusions about {x}"],
"extraction":["extract names dates and organizations from {x}","identify entities mentioned in {x}","pull structured facts out of {x}","turn {x} into structured entities and dates","parse people organizations and places from {x}"],
"device_action":["open the {x} application","create a folder named {x}","show files in {x}","launch the application {x}","list directory contents for {x}"],
"unknown_ambiguous":["help with {x}","tell me something about {x}","what do you think of {x}","can you handle this thing involving {x}","assist me somehow regarding {x}"]}
TEST={
"investigation":["conduct due diligence on {x}","assemble a dossier from lawful public data about {x}"],"search":["locate recent open information concerning {x}","retrieve public articles on {x}"],"evidence_analysis":["weigh corroboration and uncertainty for {x}","grade the support behind {x}"],"comparison":["analyze similarities and differences: {x}, {y}","which is better supported, {x} or {y}"],"timeline":["reconstruct the sequence of events for {x}","order dated events linked with {x}"],"relationship_analysis":["construct an association graph for {x}","trace organizational links from {x} to {y}"],"location_analysis":["geolocate this authorized public event involving {x}","list known public places tied to {x}"],"source_verification":["audit provenance and credibility for {x}","rate the trustworthiness of the cited source on {x}"],"contradiction_analysis":["detect inconsistent statements regarding {x}","identify mutually exclusive assertions about {x}"],"summarization":["condense the findings on {x}","brief me on the main conclusions about {x}"],"extraction":["turn {x} into structured entities and dates","parse people organizations and places from {x}"],"device_action":["launch {x}","list the contents of {x}"],"unknown_ambiguous":["can you handle {x}","assist me somehow with {x}"]}
SUB_TRAIN=["Project Atlas","Acme Labs","a procurement record","a climate claim","the incident report","a public company"];SUB_TEST=["Operation Meridian","Northwind Group","the audit memo","a vaccine claim"];PAIR_TRAIN=["Project Beacon","Contoso"];PAIR_TEST=["Project Solstice","Fabrikam"];TOKEN_RE=re.compile(r"[a-z0-9][a-z0-9_-]{1,}")
def tokenize(text):
 w=TOKEN_RE.findall(text.lower());return w+[f"{a}::{b}" for a,b in zip(w,w[1:])]
def rows(templates,subjects,pairs):return [{"text":t.format(x=x,y=pairs[i%len(pairs)]),"label":label} for label in LABELS for t in templates[label] for i,x in enumerate(subjects)]
def digest(data):return hashlib.sha256("\n".join(json.dumps(x,sort_keys=True) for x in data).encode()).hexdigest()
def predict(text,a):
 toks=Counter(tokenize(text));vs=max(1,len(a["vocabulary"]));total=sum(a["class_documents"].values());raw={}
 for label in a["labels"]:
  prior=(a["class_documents"][label]+1)/(total+len(a["labels"]));c=a["token_counts"][label];denom=sum(c.values())+vs;raw[label]=math.log(prior)+sum(n*math.log((c.get(tok,0)+1)/denom) for tok,n in toks.items())
 return max(raw,key=raw.get)
def train(output_dir):
 random.seed(SEED);tr=rows(TRAIN,SUB_TRAIN,PAIR_TRAIN);te=rows(TEST,SUB_TEST,PAIR_TEST);docs=Counter();counts=defaultdict(Counter);vocab=set()
 for r in tr:docs[r["label"]]+=1;tok=tokenize(r["text"]);counts[r["label"]].update(tok);vocab.update(tok)
 a={"version":VERSION,"labels":list(LABELS),"seed":SEED,"model_type":"multinomial_naive_bayes_word_bigram","alpha":1.0,"class_documents":dict(docs),"token_counts":{k:dict(v) for k,v in counts.items()},"vocabulary":sorted(vocab),"provenance":[{"source":"SWEEP existing intent taxonomy","usage":"labels and task vocabulary"},{"source":"SWEEP structure-focused template generator","usage":"training examples","license":"repository license"}],"splits":{"train_sha256":digest(tr),"test_sha256":digest(te),"train_examples":len(tr),"test_examples":len(te),"immutable":True},"trained_at":datetime.now(timezone.utc).isoformat()}
 correct=sum(predict(r["text"],a)==r["label"] for r in te);report={"version":VERSION,"seed":SEED,"baseline_majority_accuracy":max(Counter(r["label"] for r in te).values())/len(te),"held_out":{"examples":len(te),"correct":correct,"accuracy":correct/len(te)},"scope":"intent routing only; not general reasoning accuracy","external_data_status":"FEVEROUS reviewed but not imported because data terms/files were not mounted"};output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True);(output_dir/"intelligence_router.json").write_text(json.dumps(a,indent=2,sort_keys=True),encoding="utf-8");(output_dir/"intelligence_router_metrics.json").write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8");return report
def main():
 p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,default=Path(__file__).parent/"artifacts");args=p.parse_args();print(json.dumps(train(args.output_dir),indent=2))
if __name__=="__main__":main()
