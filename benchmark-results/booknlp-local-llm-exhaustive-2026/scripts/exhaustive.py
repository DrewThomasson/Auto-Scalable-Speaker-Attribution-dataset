#!/usr/bin/env python3
"""Resumable exhaustive, token-enumerated BookNLP task evaluation.

Gold data is accessed only by the scorer. Every input to Ollama is made from
the token stream and fixed task instructions; decision caches are keyed from
the full prompt, model digest, pass, and task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

EXP = Path(os.environ.get("BOOKNLP_EXPERIMENT_ROOT", "/home/drew/booknlp_llm_experiment")).resolve()
sys.path.insert(0, str(EXP / "src"))
from booknlp_experiment.benchmark import (  # noqa: E402
    SUPERSENSES, load_docs, select_task_split, metric_for, clean_preds,
    SCHEMAS, resource_snapshot,
)
from booknlp_experiment.data import save_semcor_normalized  # noqa: E402
from booknlp_experiment.coref_eval import score_conll  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
OUT = EXP / "results/exhaustive"
CACHE = OUT / "cache"
PRED = OUT / "predictions"
OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True); PRED.mkdir(parents=True, exist_ok=True)

LABELS = {
    # Negative/none is index zero for every sparse label space to avoid a
    # systematic positive-label preference caused by always choosing class 0.
    "events": ["NO", "EVENT"], "entities": ["NO", "START"],
    "quotes": ["NO", "OPEN", "CLOSE", "BOTH"],
    "coref": ["NO", "START"], "supersenses": ["NONE", *SUPERSENSES],
}
FIELD = {"events":"decisions", "entities":"decisions", "quotes":"decisions", "coref":"decisions", "supersenses":"decisions"}

def schema(labels, count):
    return {"type":"object", "properties":{"labels":{"type":"array", "minItems":count,"maxItems":count,"items":{"type":"integer", "enum":list(range(len(labels)))}}},
            "required":["labels"], "additionalProperties":False}

def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

@lru_cache(maxsize=None)
def digest_model(model):
    try:
        tags=json.load(urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5))
        return next((x.get("digest") for x in tags.get("models",[]) if x.get("name")==model), None)
    except Exception: return None

def call(model, task, doc, ids, pass_id, wider=False, context_hint=""):
    tokens=doc["tokens"]
    radius=48 if wider else 18
    lo=max(0,min(ids)-radius); hi=min(len(tokens),max(ids)+radius+1)
    excerpt=" ".join(f"[{j}] {tokens[j]}" for j in range(lo,hi))
    rules={
      "events":"Label EVENT only when this exact token is a lexical trigger of a specific, asserted realis event in the narrative. Exclude hypothetical, intended, future, counterfactual, generic, negated/non-asserted, and non-event uses. Label each candidate independently.",
      "entities":"Label START if a human or non-human entity mention begins exactly at this token. Include nested and common noun mentions; exclude pronouns. Label each token independently.",
      "quotes":"Label OPEN for a token beginning direct quoted speech, CLOSE for a token ending direct quoted speech, BOTH if both, and NO otherwise. Include opening/closing delimiters when they are tokenized. Exclude scare quotes, titles, and mentioned terms. Consider punctuation and dialogue dashes; label each candidate independently.",
      "coref":"Label START if a referential entity mention (including pronouns and singleton mentions) begins exactly here. Include nested noun phrases. Exclude non-referential, event, and clause spans. Label each token independently.",
      "supersenses":"For this exact token, choose its contextual WordNet 3.0 lexicographer supersense if it is a content noun or verb; otherwise NONE. Use one of the listed labels only. Label each candidate independently.",
    }[task]
    mode=("independent adversarial verification" if pass_id==1 else "initial independent judgment")
    candidate_text="; ".join(f"[{i}] {tokens[i]}" for i in ids)
    prompt=(f"You perform exhaustive literary annotation. Task={task}. Pass={mode}. {rules}\n"
            "The program has exhaustively enumerated every candidate token ID. Decide each listed candidate independently from its exact word and context. Return a compact JSON integer array named labels in exactly the same order as CANDIDATE IDS, with one label index per ID and no omissions. Do not reveal chain of thought.\n"
            f"Index-to-label map: {json.dumps(dict(enumerate(LABELS[task])))}.\n{context_hint}\nCANDIDATE IDS: {json.dumps(ids)}\nTARGET TOKEN(S): {candidate_text}\nLOCAL CONTEXT TOKENS:\n{excerpt}")
    sch=schema(LABELS[task],len(ids))
    key=hashlib.sha256(json.dumps([model,digest_model(model),task,doc["doc_id"],ids,pass_id,wider,prompt,sch],ensure_ascii=False).encode()).hexdigest()
    cp=CACHE/(key+".json")
    if cp.exists(): return json.loads(cp.read_text()), True, 0.0
    body={"model":model,"messages":[{"role":"user","content":prompt}],"stream":False,"format":sch,
          "options":{"temperature":0,"num_ctx":8192,"num_predict":max(128, len(ids)*8+64)},"think":False,"keep_alive":"10m"}
    req=urllib.request.Request("http://127.0.0.1:11434/api/chat",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    started=time.time()
    last=None
    last_content=None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req,timeout=1800) as res: response=json.loads(res.read())
            last_content=response.get("message",{}).get("content","")
            data=json.loads(response["message"]["content"])
            labels=data["labels"]
            found=validate_labels(ids,labels,task)
            result={"decisions":found,"valid":True,"prompt_tokens":response.get("prompt_eval_count"),"output_tokens":response.get("eval_count"),"eval_seconds":response.get("eval_duration",0)/1e9,"wall_seconds":time.time()-started,"attempts":attempt+1,"raw_error":None}
            atomic_json(cp,result); return result,False,time.time()-started
        except Exception as exc:
            last=repr(exc); time.sleep(1+attempt)
    result={"decisions":[{"token_id":i,"label":LABELS[task][0],"confidence":0.0} for i in ids],"valid":False,"prompt_tokens":None,"output_tokens":None,"eval_seconds":0,"wall_seconds":time.time()-started,"attempts":3,"raw_error":last,"last_content":last_content}
    atomic_json(cp,result); return result,False,time.time()-started

def batches(n, size=1):
    return [list(range(i,min(n,i+size))) for i in range(0,n,size)]

def validate_labels(ids, labels, task):
    """Map compact position-ordered model labels back to immutable token IDs."""
    if not isinstance(labels,list) or len(labels)!=len(ids):
        raise ValueError(f"expected exactly {len(ids)} ordered labels, received {len(labels) if isinstance(labels,list) else 'non-array'}")
    if any(type(x) is not int or x not in range(len(LABELS[task])) for x in labels):
        raise ValueError("one or more decisions contains an invalid label index")
    return [{"token_id":i,"label":LABELS[task][v],"confidence":0.5} for i,v in zip(ids,labels)]

def collect(model, task, doc):
    n=len(doc["tokens"]); ids=list(range(n)); all_passes=[]; stats={"calls":0,"cache_hits":0,"prompt_tokens":0,"output_tokens":0,"wall_seconds":0,"model_wall_seconds":0,"invalid_batches":0}
    invalid_ids=set()
    for pass_id in (0,1):
        # Opposite traversal reduces sequence-position / recency artifacts between votes.
        groups=batches(n)
        if pass_id: groups=list(reversed(groups))
        votes={}
        for group in groups:
            r,cached,elapsed=call(model,task,doc,group,pass_id,wider=bool(pass_id))
            stats["calls"]+=1; stats["cache_hits"]+=int(cached); stats["wall_seconds"]+=elapsed
            stats["model_wall_seconds"]+=r.get("wall_seconds") or 0
            stats["prompt_tokens"]+=r.get("prompt_tokens") or 0; stats["output_tokens"]+=r.get("output_tokens") or 0
            stats["invalid_batches"]+=int(not r["valid"])
            if not r["valid"]: invalid_ids.update(group)
            for x in r["decisions"]: votes[int(x["token_id"])]=x
            if group[0] % 100 == 0 or group[-1] == n-1:
                print(f"{model} {task} {doc['doc_id']} pass={pass_id+1} candidates={group[0]}-{group[-1]} cache={int(cached)}",flush=True)
        all_passes.append(votes)
    # Resolve disagreement with a third structured adjudication call, passing both labels.
    disagree=[i for i in ids if all_passes[0][i]["label"]!=all_passes[1][i]["label"]]
    resolved=dict(all_passes[0]); confidence={i:(0.0 if i in invalid_ids else (1.0 if i not in disagree else 0.0)) for i in ids}
    for group in batches(len(disagree),32):
        sub=disagree[group[0]:group[-1]+1]
        if not sub: continue
        # third vote is independently queried with contrastive instruction and longer context
        ids_before=list(sub)
        r,cached,elapsed=call(model,task,doc,ids_before,2,wider=True,
          context_hint="Make a fresh decision; neither prior response is privileged. Resolve genuine ambiguity conservatively.")
        stats["calls"]+=1; stats["cache_hits"]+=int(cached); stats["wall_seconds"]+=elapsed
        stats["model_wall_seconds"]+=r.get("wall_seconds") or 0
        stats["prompt_tokens"]+=r.get("prompt_tokens") or 0; stats["output_tokens"]+=r.get("output_tokens") or 0; stats["invalid_batches"]+=int(not r["valid"])
        for x in r["decisions"]:
            i=int(x["token_id"]); labs=[all_passes[0][i]["label"],all_passes[1][i]["label"],x["label"]]
            counts={lab:labs.count(lab) for lab in set(labs)}
            resolved[i]={**x,"label":max(counts,key=lambda k:(counts[k], -LABELS[task].index(k)))}
            confidence[i]=max(counts.values())/3
        if not r["valid"]:
            for i in ids_before: confidence[i]=0.0
    for i in invalid_ids: confidence[i]=0.0
    vote2={}
    for i in ids:
        a,b=all_passes[0][i]["label"],all_passes[1][i]["label"]
        vote2[i]={**all_passes[1][i],"label":a if a==b else LABELS[task][0]}
    layers={name:{str(i):x["label"] for i,x in layer.items()} for name,layer in (("pass1",all_passes[0]),("pass2",all_passes[1]),("two_pass_vote",vote2),("final",resolved))}
    return resolved,confidence,stats,layers

def question(model, task, doc, prompt, sch, stage, ident):
    key=hashlib.sha256(json.dumps(["question-v3",model,digest_model(model),task,doc["doc_id"],stage,ident,prompt,sch],ensure_ascii=False).encode()).hexdigest(); p=CACHE/(key+".json")
    if p.exists(): return json.loads(p.read_text()),True
    body={"model":model,"messages":[{"role":"user","content":prompt}],"stream":False,"format":sch,"options":{"temperature":0,"num_ctx":8192,"num_predict":256},"think":False}
    req=urllib.request.Request("http://127.0.0.1:11434/api/chat",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    last=None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req,timeout=1800) as r: response=json.loads(r.read())
            value=json.loads(response["message"]["content"])
            if not isinstance(value,dict) or set(value)!=set(sch.get("required",[])):
                raise ValueError("response keys do not satisfy schema")
            if stage.startswith("pair-") and type(value.get("same")) is not bool:
                raise ValueError("pairwise identity answer is not boolean")
            if stage.startswith("speaker"):
                arr=value.get("speakers")
                if not isinstance(arr,list) or len(arr)!=1: raise ValueError("speaker response must contain exactly one quote decision")
                item=arr[0]
                needed={"quote_start_token","quote_end_token","speaker_start_token","speaker_end_token","speaker_cluster"}
                if not isinstance(item,dict) or set(item)!=needed: raise ValueError("speaker item does not match required schema")
            if stage in {"entity-span","mention-span"}:
                if type(value.get("end_token")) is not int or not isinstance(value.get("type"),str): raise ValueError("span resolution has invalid end/type")
            if stage=="entity-verify" and type(value.get("yes")) is not bool:
                raise ValueError("span verification answer is not boolean")
            value.update({"_usage":{"prompt_tokens":response.get("prompt_eval_count",0),"output_tokens":response.get("eval_count",0),"wall_seconds":response.get("total_duration",0)/1e9},"_valid":True,"_attempts":attempt+1})
            atomic_json(p,value); return value,False
        except Exception as e:
            last=repr(e); time.sleep(1+attempt)
    value={"error":last,"_valid":False,"_attempts":3,"_usage":{"prompt_tokens":0,"output_tokens":0,"wall_seconds":0}}
    atomic_json(p,value); return value,False

def record_question(stats,value,cached):
    stats["calls"]+=1; stats["cache_hits"]+=int(cached)
    usage=value.get("_usage",{}) if isinstance(value,dict) else {}
    for k in ("prompt_tokens","output_tokens","wall_seconds"):
        stats[k]=stats.get(k,0)+(usage.get(k) or 0)
    stats["model_wall_seconds"]=stats.get("model_wall_seconds",0)+(usage.get("wall_seconds") or 0)
    stats["invalid_followups"]=stats.get("invalid_followups",0)+int(value.get("_valid") is False)

def entity_spans(model,doc,decisions,conf,stats):
    starts=[i for i,x in decisions.items() if x["label"]=="START"]
    types=["PER","FAC","GPE","LOC","VEH","ORG"]
    sch={"type":"object","properties":{"end_token":{"type":"integer"},"type":{"type":"string","enum":types},"confidence":{"type":"number"}},"required":["end_token","type","confidence"],"additionalProperties":False}
    out=[]; toks=doc["tokens"]
    for i in starts:
        lo=max(0,i-18);hi=min(len(toks),i+22)
        prompt=(f"A possible entity mention starts at token {i}. In this local context, choose the shortest complete human/non-human entity noun phrase end token and type (PER person, FAC facility, GPE geopolitical entity, LOC location, VEH vehicle, ORG organization). Nested entities allowed. If no valid span begins here, set end_token=-1. No pronouns.\n"+" ".join(f"[{j}] {toks[j]}" for j in range(lo,hi)))
        x,cached=question(model,"entities",doc,prompt,sch,"entity-span",i); record_question(stats,x,cached)
        end=x.get("end_token");typ=x.get("type")
        if type(end) is int and i<=end<len(toks) and typ in types:
            verify=(f"Verify exact entity span [{i}:{end}] ({' '.join(toks[i:end+1])!r}) in context. Is it a complete human/non-human entity mention of type {typ}, with the shortest complete span? Return yes/no and confidence.\n"+" ".join(f"[{j}] {toks[j]}" for j in range(lo,hi)))
            vs={"type":"object","properties":{"yes":{"type":"boolean"},"confidence":{"type":"number"}},"required":["yes","confidence"],"additionalProperties":False}
            y,cache2=question(model,"entities",doc,verify,vs,"entity-verify",i); record_question(stats,y,cache2)
            c=(float(conf.get(i,.5))+float(x.get("confidence",.5))+float(y.get("confidence",.5)))/3
            if y.get("yes"): out.append({"start_token":i,"end_token":end,"type":typ,"confidence":c})
    return out

def quote_spans(doc,decisions,conf):
    toks=doc["tokens"]; out=[]; opened=None
    for i in range(len(toks)):
        lab=decisions[i]["label"]
        if lab in ("OPEN","BOTH"):
            if opened is not None and opened<i: out.append({"start_token":opened,"end_token":i,"confidence":min(conf[opened],conf[i])})
            opened=i
        if lab in ("CLOSE","BOTH"):
            if opened is None: continue
            if i>=opened: out.append({"start_token":opened,"end_token":i,"confidence":min(conf[opened],conf[i])})
            opened=None
    if opened is not None: out.append({"start_token":opened,"end_token":len(toks)-1,"confidence":conf[opened]})
    return out

def predict_speakers(model,doc,quotes,stats):
    toks=doc["tokens"]; out=[]; clusters={}; sch=SCHEMAS["speakers"]
    for q in quotes:
        a,b=q["start_token"],q["end_token"]
        answers=[]
        for pass_id,radius in enumerate((45,140)):
            lo=max(0,a-radius);hi=min(len(toks),b+radius+1)
            focus=("First inspect local speech tags and turn position." if pass_id==0 else "Independently re-evaluate with broader forward and backward context; check addressee-versus-speaker, nearby actions, and turn-taking.")
            prompt=(f"{focus} Identify who speaks the exact direct quotation [{a},{b}]. Return the speaker's visible mention span if supported, otherwise null. Use a consistent surface-based character cluster ID across this document. Do not guess the addressee. Return exactly one item.\n"+f"QUOTE: "+" ".join(f"[{j}] {toks[j]}" for j in range(lo,hi)))
            x,cached=question(model,"speakers",doc,prompt,sch,f"speaker-pass-{pass_id}",f"{a}-{b}");record_question(stats,x,cached)
            arr=x.get("speakers",[]) if isinstance(x,dict) else []
            answers.append(arr[0] if arr else {"speaker_start_token":None,"speaker_end_token":None,"speaker_cluster":None})
        def identity(s):
            a1,b1=s.get("speaker_start_token"),s.get("speaker_end_token")
            if type(a1) is int and type(b1) is int: return ("span",a1,b1)
            return ("cluster",str(s.get("speaker_cluster") or "").casefold())
        if identity(answers[0])==identity(answers[1]):
            chosen=answers[0]; conf=1.0
        else:
            lo=max(0,a-400);hi=min(len(toks),b+401)
            prompt=("Adjudicate speaker identity for this quotation after independently considering the wider scene. Choose the most likely actual speaker; distinguish speaker from addressee. Null is allowed only if evidence is insufficient. Return exactly one item.\n"+f"QUOTE [{a},{b}]: "+" ".join(f"[{j}] {toks[j]}" for j in range(lo,hi)))
            x,cached=question(model,"speakers",doc,prompt,sch,"speaker-adjudication",f"{a}-{b}");record_question(stats,x,cached)
            arr=x.get("speakers",[]) if isinstance(x,dict) else []
            third=arr[0] if arr else {"speaker_start_token":None,"speaker_end_token":None,"speaker_cluster":None}
            choices=[identity(s) for s in (*answers,third)]; best=max(set(choices),key=lambda z:(choices.count(z),str(z)))
            chosen=next(s for s in (third,*answers) if identity(s)==best);conf=choices.count(best)/3
        s=chosen
        label=s.get("speaker_cluster")
        if label is not None: clusters.setdefault(label,f"c{len(clusters)+1}")
        out.append({"quote_start_token":a,"quote_end_token":b,"speaker_start_token":s.get("speaker_start_token"),"speaker_end_token":s.get("speaker_end_token"),"speaker_cluster":clusters.get(label) if label else None,"confidence":conf})
    return out

def cluster_coref(model,doc,mentions,stats):
    """Pairwise antecedent checks plus constrained union-find consistency.
    Candidate antecedents include all repeated surfaces and up to 20 recent spans.
    """
    toks=doc["tokens"]; n=len(mentions); parent=list(range(n)); edge_conf={};
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[max(ra,rb)]=min(ra,rb)
    sch={"type":"object","properties":{"same":{"type":"boolean"},"confidence":{"type":"number"}},"required":["same","confidence"],"additionalProperties":False}
    surfaces=[" ".join(toks[m["start_token"]:m["end_token"]+1]).lower() for m in mentions]
    pairs=[]
    for j,m in enumerate(mentions):
        recent=list(range(max(0,j-20),j))
        # Long-distance exact matching is useful for multi-token names; allowing
        # every occurrence of a one-token pronoun as an antecedent explodes pairs.
        multi_token=(mentions[j]["end_token"]-mentions[j]["start_token"]+1)>1
        exact=[i for i in range(j) if surfaces[i]==surfaces[j] and multi_token and surfaces[j]]
        for i in sorted(set(recent+exact)):
            if mentions[i].get("type")!=m.get("type"): continue
            pairs.append((i,j))
    byright={}
    for i,j in pairs: byright.setdefault(j,[]).append(i)
    disagreements=0
    for j,ants in byright.items():
        # pairwise atomic candidate comparisons, independently worded twice
        for i in ants:
            a,b=mentions[i],mentions[j]
            t1=" ".join(toks[a['start_token']:a['end_token']+1]); t2=" ".join(toks[b['start_token']:b['end_token']+1])
            lo=max(0,a['start_token']-80); hi=min(len(toks),b['end_token']+81)
            context=" ".join(f"[{k}] {toks[k]}" for k in range(lo,hi))
            yes=[]; cf=[]
            for pass_id in (0,1):
                prompt=(f"Do mention A [{a['start_token']}:{a['end_token']}] {t1!r} and mention B [{b['start_token']}:{b['end_token']}] {t2!r} refer to the same underlying entity? Decide identity, not similarity. Pass {pass_id+1}; be conservative and consider intervening referents.\n{context}")
                x,cached=question(model,"coref",doc,prompt,sch,f"pair-{pass_id}",f"{i}-{j}");record_question(stats,x,cached)
                yes.append(bool(x.get("same")));cf.append(float(x.get("confidence",0)))
            if yes[0]!=yes[1]:
                disagreements+=1
                prompt=(f"Resolve a disagreement: do mention A [{a['start_token']}:{a['end_token']}] {t1!r} and mention B [{b['start_token']}:{b['end_token']}] {t2!r} refer to the same underlying entity? Judge identity conservatively using the entire supplied context.\n{context}")
                x,cached=question(model,"coref",doc,prompt,sch,"pair-resolve",f"{i}-{j}");record_question(stats,x,cached)
                yes.append(bool(x.get("same")))
            edge_conf[(i,j)]=sum(yes)/len(yes)
    # Global consistency: a positive edge cannot merge components if a directly
    # queried negative edge would then lie inside the resulting cluster.
    conflicts=0
    negative=[(i,j,w) for (i,j),w in edge_conf.items() if w<.5]
    for i,j,w in sorted(((i,j,w) for (i,j),w in edge_conf.items() if w>.5),key=lambda x:(-x[2],x[0],x[1])):
        ri,rj=find(i),find(j)
        if ri==rj: continue
        blocked=False
        for a,b,nw in negative:
            ra,rb=find(a),find(b)
            if (ra==ri and rb==rj) or (ra==rj and rb==ri): blocked=True;break
        if blocked: conflicts+=1
        else: union(i,j)
    stats["coref_pair_disagreements"]=disagreements
    stats["coref_positive_edges_blocked_by_global_negative_constraint"]=conflicts
    groups={}
    for i in range(n):groups.setdefault(find(i),[]).append(i)
    ordered=sorted(groups.values(),key=lambda g:min(mentions[i]["start_token"] for i in g)); labels={i:f"c{k+1}" for k,g in enumerate(ordered) for i in g}
    for i,m in enumerate(mentions):m["cluster"]=labels[i]
    return mentions,edge_conf

def run_doc(model,task,doc,split,protocol):
    predfile=PRED/f"{model.replace(':','_')}_{split}_{task}.jsonl"
    runner_fingerprint=hashlib.sha256(Path(__file__).read_bytes()+str(digest_model(model)).encode()+task.encode()).hexdigest()
    old={}
    if predfile.exists():
        for line in predfile.read_text().splitlines():
            try:r=json.loads(line);old[r["doc_id"]]=r
            except Exception:pass
    if doc["doc_id"] in old and old[doc["doc_id"]].get("runner_fingerprint")==runner_fingerprint: return old[doc["doc_id"]]
    old.pop(doc["doc_id"],None)
    started=time.time(); resources_before=resource_snapshot(); decision_task="quotes" if task=="speakers" else task
    decisions,conf,stats,layers=collect(model,decision_task,doc)
    raw=[]
    if task=="events":
        raw=[{"start_token":i,"end_token":i,"confidence":conf[i]} for i,x in decisions.items() if x["label"]=="EVENT"]
    elif task=="entities": raw=entity_spans(model,doc,decisions,conf,stats)
    elif task=="quotes": raw=quote_spans(doc,decisions,conf)
    elif task=="supersenses":
        raw=[{"start_token":i,"end_token":i,"label":x["label"],"confidence":conf[i]} for i,x in decisions.items() if x["label"]!="NONE"]
    elif task=="coref":
        starts=[i for i,x in decisions.items() if x["label"]=="START"]
        raw=[]; ts=doc["tokens"]
        for i in starts:
            lo=max(0,i-12);hi=min(len(ts),i+18)
            sp={"type":"object","properties":{"end_token":{"type":"integer"},"type":{"type":"string","enum":["PER","FAC","GPE","LOC","VEH","ORG"]},"confidence":{"type":"number"}},"required":["end_token","type","confidence"],"additionalProperties":False}
            prompt=f"Resolve the shortest referring entity NP/pronoun span beginning at token {i} and type; return -1 if not a referring mention.\n"+" ".join(f"[{j}] {ts[j]}" for j in range(lo,hi))
            x,cached=question(model,task,doc,prompt,sp,"mention-span",i);record_question(stats,x,cached)
            if type(x.get("end_token")) is int and i<=x["end_token"]<len(ts) and x.get("type") in sp["properties"]["type"]["enum"]:
                raw.append({"start_token":i,"end_token":x["end_token"],"type":x["type"],"confidence":(conf[i]+float(x.get("confidence",.5)))/2})
        raw,edges=cluster_coref(model,doc,raw,stats)
    elif task=="speakers":
        # Speaker-only cohort is conditioned on quotes produced by this same exhaustive quote detector.
        quotes=quote_spans(doc,decisions,conf); raw=predict_speakers(model,doc,quotes,stats)
    # benchmark predictions omit non-schema confidence metadata; preserve it separately for teacher scoring.
    clean=clean_preds(task, {"entities":"entities","events":"events","coref":"mentions","quotes":"quotes","speakers":"speakers","supersenses":"supersenses"}[task] and ({"entities":raw} if task=="entities" else {"events":raw} if task=="events" else {"mentions":raw} if task=="coref" else {"quotes":raw} if task=="quotes" else {"speakers":raw} if task=="speakers" else {"supersenses":raw}),len(doc["tokens"]))
    record={"doc_id":doc["doc_id"],"task":task,"prediction":clean,"teacher_candidates":raw,"token_confidence":{str(k):v for k,v in conf.items()},"decision_layers":layers,"stats":stats,"wall_seconds":time.time()-started,"resources_before":resources_before,"resources_after":resource_snapshot(),"method":"exhaustive-token-enumeration-v1","model":model,"model_digest":digest_model(model),"runner_fingerprint":runner_fingerprint}
    # JSONL checkpoint replaced atomically after every complete document.
    old[doc["doc_id"]]=record; atomic_json(predfile, None) if False else None
    tmp=predfile.with_suffix(".tmp"); tmp.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in old.values()),encoding="utf-8");tmp.replace(predfile)
    return record

def teacher_curve(task,rows):
    """Post-hoc precision/recall as a fraction of predicted labels is retained."""
    if task=="speakers":
        items=sorted(((float(x.get("confidence",0)),d["doc_id"],x) for d,r in rows for x in r["teacher_candidates"]),key=lambda x:(-x[0],x[1],x[2]["quote_start_token"]))
        points=[]
        for fraction in (.25,.50,.75,.90,1.0):
            chosen=items[:round(len(items)*fraction)]; by={d["doc_id"]:[] for d,_ in rows}
            for _,docid,item in chosen: by[docid].append(item)
            scores=[metric_for("speakers",d["gold"],by[d["doc_id"]]) for d,_ in rows]
            points.append({"fraction_of_model_predictions_retained":fraction,"selected_quotes":len(chosen),"macro_conditional_speaker_B3_F1":sum(s["speaker_B3"]["f1"] for s in scores)/len(scores) if scores else 0,"macro_joint_quote_speaker_B3_F1":sum(s["joint_quote_speaker_B3_f1"] for s in scores)/len(scores) if scores else 0})
        return {"available":True,"confidence_source":"independent context-view agreement; uncalibrated","scope":"conditional and joint quote-speaker B3 under the established metric","points":points}
    keyfn={"entities":lambda x:(x["start_token"],x["end_token"],x.get("type")),"events":lambda x:(x["start_token"],x["end_token"]),"coref":lambda x:(x["start_token"],x["end_token"]),"quotes":lambda x:(x["start_token"],x["end_token"]),"supersenses":lambda x:(x["start_token"],x["end_token"],x.get("label"))}[task]
    items=[]; gold=set()
    for doc,record in rows:
        if task=="events":
            gold.update((doc["doc_id"],i) for x in doc["gold"]["events"] for i in range(x["start_token"],x["end_token"]+1))
        else:
            gold.update((doc["doc_id"],*keyfn(x)) for x in doc["gold"]["coref" if task=="coref" else task])
        seen=set()
        for x in record["teacher_candidates"]:
            k=keyfn(x)
            if k in seen: continue
            seen.add(k); items.append((float(x.get("confidence",0)),str(doc["doc_id"]),k))
    items.sort(key=lambda x:(-x[0],x[1],x[2]))
    points=[]
    for fraction in (.25,.50,.75,.90,1.0):
        n=round(len(items)*fraction); chosen=items[:n]
        matched=sum(((docid,k[0]) in gold if task=="events" else (docid,*k) in gold) for _,docid,k in chosen)
        precision=matched/n if n else None; recall=matched/len(gold) if gold else 0.0
        points.append({"fraction_of_model_predictions_retained":fraction,"selected_predictions":n,"available_predictions":len(items),"precision":precision,"gold_recall":recall,"f1":(2*precision*recall/(precision+recall) if precision is not None and precision+recall else 0.0)})
    return {"available":True,"confidence_source":"vote agreement and/or local model confidence; uncalibrated", "scope":"coref measures mention spans only, not cluster identity" if task=="coref" else "exact task labels","points":points}

def ablation_scores(task,rows):
    """Score saved first judgment, two-pass vote, and disagreement-resolved output."""
    source_task="quotes" if task=="speakers" else task
    layers=("pass1","two_pass_vote","final")
    output={}
    for layer in layers:
        per=[]
        for doc,record in rows:
            vals=record.get("decision_layers",{}).get(layer,{})
            decisions={int(k):{"label":v} for k,v in vals.items()}
            if source_task=="events": pred=[{"start_token":i,"end_token":i} for i,x in decisions.items() if x["label"]=="EVENT"]
            elif source_task=="supersenses": pred=[{"start_token":i,"end_token":i,"label":x["label"]} for i,x in decisions.items() if x["label"]!="NONE"]
            elif source_task=="quotes": pred=quote_spans(doc,decisions,{i:1.0 for i in decisions})
            elif source_task in {"entities","coref"}:
                gold=doc["gold"]["entities" if source_task=="entities" else "coref"]
                gs={int(x["start_token"]) for x in gold}; ps={i for i,x in decisions.items() if x["label"]=="START"}
                tp=len(gs&ps); p=tp/len(ps) if ps else 0;r=tp/len(gs) if gs else 0
                per.append({"precision":p,"recall":r,"f1":2*p*r/(p+r) if p+r else 0,"matched":tp,"predicted":len(ps),"gold":len(gs)}); continue
            else: continue
            per.append(metric_for(source_task,doc["gold"],pred))
        if per:
            if source_task in {"entities","coref"}:
                tp=sum(x["matched"] for x in per);pn=sum(x["predicted"] for x in per);gn=sum(x["gold"] for x in per);p=tp/pn if pn else 0;r=tp/gn if gn else 0
                output[layer]={"mention_start_micro":{"precision":p,"recall":r,"f1":2*p*r/(p+r) if p+r else 0,"matched":tp,"predicted":pn,"gold":gn}}
            else:
                tp=sum(x["matched"] for x in per);pn=sum(x["predicted"] for x in per);gn=sum(x["gold"] for x in per);p=tp/pn if pn else 0;r=tp/gn if gn else 0
                output[layer]={"pooled_micro":{"precision":p,"recall":r,"f1":2*p*r/(p+r) if p+r else 0,"matched":tp,"predicted":pn,"gold":gn}}
    return output

def main():
    global EXP
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True);ap.add_argument("--task",choices=["entities","events","coref","quotes","speakers","supersenses"],required=True)
    ap.add_argument("--split",choices=["validation","test"],default="validation");ap.add_argument("--limit",type=int,default=None)
    ap.add_argument("--ids",nargs="*");args=ap.parse_args()
    if args.task=="supersenses":
        p=EXP/"datasets/normalized/semcor.jsonl"
        if not p.exists(): save_semcor_normalized(p)
        docs=[json.loads(x) for x in p.read_text().splitlines()]
    else: docs=load_docs()
    selected,protocol=select_task_split(args.task,docs,args.split)
    if args.ids: selected=[d for d in selected if d["doc_id"] in set(args.ids)]
    if args.limit: selected=selected[:args.limit]
    rows=[]
    for doc in selected:
        r=run_doc(args.model,args.task,doc,args.split,protocol); rows.append((doc,r))
        print(f"DONE {args.task} {doc['doc_id']} tokens={len(doc['tokens'])} calls={r['stats']['calls']} sec={r['wall_seconds']:.1f}",flush=True)
    # Emit genuine scorer metrics; pool counts only across document-local scores.
    scores=[metric_for(args.task,d["gold"],r["prediction"]) for d,r in rows]
    if args.task in {"entities","events","quotes","supersenses"}:
        matched=sum(s["matched"] for s in scores);pred=sum(s["predicted"] for s in scores);gold=sum(s["gold"] for s in scores)
        p=matched/pred if pred else 0;r=matched/gold if gold else 0; metrics={"precision":p,"recall":r,"f1":2*p*r/(p+r) if p+r else 0,"matched":matched,"predicted":pred,"gold":gold}
    elif args.task=="coref":
        by={r["doc_id"]:r["prediction"] for _,r in rows}; metrics=score_conll([d for d,_ in rows],by,OUT/"coref_scorer"/args.model.replace(":","_")/args.split)
        metrics.pop("raw_output",None)
    else: metrics={"macro_joint_speaker_B3_F1":sum(s["joint_quote_speaker_B3_f1"] for s in scores)/len(scores) if scores else 0,"macro_conditional_speaker_B3_F1":sum(s["speaker_B3"]["f1"] for s in scores)/len(scores) if scores else 0}
    out_tokens=sum(r["stats"]["output_tokens"] for _,r in rows); process_wall=sum(r["wall_seconds"] for _,r in rows); inference_wall=sum(r["stats"]["model_wall_seconds"] for _,r in rows)
    ram_values=[r.get("resources_after",{}).get("system_ram_used_mib") for _,r in rows if r.get("resources_after",{}).get("system_ram_used_mib") is not None]
    gpu_values=[r.get("resources_after",{}).get("gpu_used_mib") for _,r in rows if r.get("resources_after",{}).get("gpu_used_mib") is not None]
    summary={"model":args.model,"task":args.task,"split":args.split,"documents":len(rows),"token_total":sum(len(d["tokens"]) for d,_ in rows),"method":"exhaustive-token-enumeration-v1","runtime_settings":{"ollama":"0.34.3","temperature":0,"thinking":False,"context_tokens":8192,"candidate_tokens_per_request":1,"num_predict_per_atomic_call":128},"model_digest":digest_model(args.model),"runner_fingerprint":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"metrics":metrics,"ablation_metrics":ablation_scores(args.task,rows),"teacher_mode":teacher_curve(args.task,rows),"per_document":{d["doc_id"]:s for (d,_),s in zip(rows,scores)},"inference":{"calls":sum(r["stats"]["calls"] for _,r in rows),"cache_hits":sum(r["stats"]["cache_hits"] for _,r in rows),"prompt_tokens":sum(r["stats"]["prompt_tokens"] for _,r in rows),"output_tokens":out_tokens,"output_tokens_per_model_second":out_tokens/inference_wall if inference_wall else 0,"model_wall_seconds":inference_wall,"wall_seconds":inference_wall,"latest_process_wall_seconds":process_wall,"invalid_batches":sum(r["stats"]["invalid_batches"] for _,r in rows),"invalid_followups":sum(r["stats"].get("invalid_followups",0) for _,r in rows),"peak_gpu_used_mib":max(gpu_values) if gpu_values else None,"peak_system_ram_used_mib":max(ram_values) if ram_values else None},"protocol":protocol,"gold_in_prompts":False,"confidence":"un-calibrated vote agreement; selective scores are descriptive, not calibrated probabilities"}
    suffix=f"_{args.split}"+(f"_pilot{len(rows)}" if args.limit or args.ids else "")+f"_{args.task}.json"
    op=OUT/"summaries"/f"{args.model.replace(':','_')}{suffix}";atomic_json(op,summary);print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__": main()
