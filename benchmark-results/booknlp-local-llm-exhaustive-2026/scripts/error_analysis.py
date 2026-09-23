#!/usr/bin/env python3
"""Create task/class failure summaries locally without exporting passage text."""
import json,os
from collections import defaultdict
from pathlib import Path
import sys
ROOT=Path(os.environ.get("BOOKNLP_EXPERIMENT_ROOT","/home/drew/booknlp_llm_experiment")).resolve()
sys.path.insert(0,str(ROOT/"src"))
from booknlp_experiment.benchmark import load_docs,select_task_split,metric_for

HERE=Path(__file__).resolve().parents[1]
PRED=ROOT/"results/exhaustive/predictions"

def prf(tp,pn,gn):
    p=tp/pn if pn else 0.;r=tp/gn if gn else 0.
    return {"precision":p,"recall":r,"f1":2*p*r/(p+r) if p+r else 0.,"matched":tp,"predicted":pn,"gold":gn}

def main():
    reports=[]
    for path in sorted(PRED.glob("*_test_*.jsonl")):
        parts=path.name.split("_"); task=parts[-1].removesuffix(".jsonl"); model="_".join(parts[:-2]).replace("_",":",1)
        if task not in {"entities","events","coref","quotes","speakers","supersenses"}: continue
        docs=load_docs()
        if task=="supersenses": docs=[json.loads(x) for x in (ROOT/"datasets/normalized/semcor.jsonl").read_text().splitlines()]
        selected,_=select_task_split(task,docs,"test"); by={d["doc_id"]:d for d in selected}
        rows={}
        for line in path.read_text().splitlines():
            try:
                r=json.loads(line)
                if r["doc_id"] in by: rows[r["doc_id"]]=r
            except Exception: pass
        classes={}; docs_out=[]
        for docid,r in rows.items():
            doc=by[docid]; pred=r["prediction"]
            if task in {"entities","supersenses"}:
                field="entities" if task=="entities" else "supersenses"; g=doc["gold"][field]
                attr="type" if task=="entities" else "label"
                labels=sorted({x[attr] for x in g}|{x[attr] for x in pred if attr in x})
                per={}
                for label in labels:
                    gs={(x["start_token"],x["end_token"]) for x in g if x[attr]==label}
                    ps={(x["start_token"],x["end_token"]) for x in pred if x.get(attr)==label}
                    per[label]=prf(len(gs&ps),len(ps),len(gs))
                    agg=classes.setdefault(label,[0,0,0]);agg[0]+=len(gs&ps);agg[1]+=len(ps);agg[2]+=len(gs)
                docs_out.append({"doc_id":docid,"per_class":per})
            elif task=="coref":
                g=doc["gold"]["coref"]; g_by_type=defaultdict(set); p={(x["start_token"],x["end_token"]) for x in pred}
                for x in g:g_by_type[x.get("mention_type","UNKNOWN")].add((x["start_token"],x["end_token"]))
                docs_out.append({"doc_id":docid,"gold_mention_type_recall":{k:(len(v&p)/len(v) if v else 0) for k,v in g_by_type.items()},"exact_mention_span_f1":prf(len(set().union(*g_by_type.values())&p),len(p),sum(len(v) for v in g_by_type.values()))})
            elif task=="speakers":
                m=metric_for("speakers",doc["gold"],pred)
                docs_out.append({"doc_id":docid,"joint_speaker_B3_F1":m.get("joint_quote_speaker_B3_f1"),"conditional_speaker_B3":m.get("speaker_B3"),"gold_quotes":m.get("gold_quotes"),"predicted_quotes":m.get("predicted_quotes"),"exact_quote_matches":m.get("exact_quote_matches")})
            else:
                docs_out.append({"doc_id":docid,"primary_metrics":r})
        class_metrics={k:prf(*v) for k,v in classes.items()}
        report={"model":model,"task":task,"split":"test","documents_available":len(rows),"documents_expected":len(selected),"complete":len(rows)==len(selected),"class_scores":class_metrics,"per_document":docs_out,"analysis_note":"Only IDs, aggregate counts, and metrics are emitted; no benchmark passage or gold text."}
        reports.append(report)
    out=ROOT/"results/exhaustive/reports/error_analysis.json";out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(reports,indent=2)+"\n")
    print(out)

if __name__=="__main__":main()
