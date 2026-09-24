#!/usr/bin/env python3
"""Build safe aggregate comparison artifacts from private local summaries."""
import csv,json,os
from pathlib import Path

ROOT=Path(os.environ.get("BOOKNLP_EXPERIMENT_ROOT","/home/drew/booknlp_llm_experiment")).resolve()
REPO=Path(__file__).resolve().parents[3]
HERE=REPO/"benchmark-results/booknlp-local-llm-exhaustive-2026"
BASE=REPO/"benchmark-results/booknlp-local-llm-2026"
SUM=ROOT/"results/exhaustive/summaries"

def f1(summary):
    """Return the established primary F1 from either summary schema.

    Exhaustive summaries store one task-specific ``metrics`` object. The
    frozen original benchmark stores pooled metrics under ``micro_metrics``
    and document-averaged metrics under ``macro_metrics``. Select the same
    aggregation used by the exhaustive scorer for a fair side-by-side row.
    """
    if not summary:
        return None
    task=summary["task"]
    if "metrics" in summary:
        m=summary["metrics"]
        if task in {"entities","events","quotes","supersenses"}: return m.get("f1")
        if task=="coref": return m.get("CoNLL_F1")
        return m.get("macro_joint_speaker_B3_F1",m.get("joint_quote_speaker_B3_f1"))
    macro=summary.get("macro_metrics",{})
    if task=="coref": return macro.get("CoNLL_F1")
    if task=="speakers": return macro.get("joint_quote_speaker_B3_f1")
    return summary.get("micro_metrics",{}).get("f1")

def model_order(name):
    return {"qwen3.5:0.8b":0,"qwen3.5:4b":1,"qwen3.5:9b":2,"gemma4:12b":3}.get(name,99)

def main():
    baseline=json.loads((BASE/"results/results_test_only.json").read_text())
    by={(r["model"],r["task"]):r for r in baseline if r["split"]=="test"}
    expected={t:int(by[("BookNLP-small",t)]["documents"]) for t in ("entities","events","coref","quotes","speakers","supersenses")}
    all_summaries=[json.loads(p.read_text()) for p in sorted(SUM.glob("*.json"))]
    local=[s for s in all_summaries if s.get("split")=="test"]
    validation=[s for s in all_summaries if s.get("split")=="validation"]
    rows=[]
    for s in local:
        task=s["task"]; model=s["model"]; old=by.get((model,task)); book=by.get(("BookNLP-small",task)); score=f1(s)
        oldscore=f1(old) if old else None; bookscore=f1(book) if book else None
        n=int(s["documents"]); total=expected[task]
        inf=s["inference"]
        rows.append({"model":model,"task":task,"documents_completed":n,"documents_expected":total,"complete":n==total,"status":"complete" if n==total else "partial","exhaustive_precision":s["metrics"].get("precision"),"exhaustive_recall":s["metrics"].get("recall"),"exhaustive_f1_or_primary":score,"old_method_f1_or_primary":oldscore,"booknlp_small_f1_or_primary":bookscore,"delta_vs_old":score-oldscore if score is not None and oldscore is not None else None,"delta_vs_booknlp":score-bookscore if score is not None and bookscore is not None else None,"inference_calls":inf["calls"],"input_tokens":inf["prompt_tokens"],"output_tokens":inf["output_tokens"],"output_tokens_per_second":inf.get("output_tokens_per_model_second"),"wall_seconds":inf["wall_seconds"],"peak_gpu_used_mib":inf.get("peak_gpu_used_mib"),"peak_system_ram_used_mib":inf.get("peak_system_ram_used_mib"),"invalid_batches":inf["invalid_batches"],"invalid_followups":inf.get("invalid_followups",0),"context_tokens":s.get("runtime_settings",{}).get("context_tokens"),"temperature":s.get("runtime_settings",{}).get("temperature"),"model_digest":s.get("model_digest"),"split":"test","method":"single-token exhaustive multi-pass"})
    rows.sort(key=lambda x:(x["task"],model_order(x["model"])))
    out=HERE/"results";out.mkdir(parents=True,exist_ok=True)
    summary_out=out/"summaries";summary_out.mkdir(parents=True,exist_ok=True)
    for s in local:
        safe_model=s["model"].replace(":","_").replace("/","_")
        filename=f"{safe_model}_{s['split']}_{s['task']}.json"
        (summary_out/filename).write_text(json.dumps(s,indent=2)+"\n",encoding="utf-8")
    (out/"comparison.json").write_text(json.dumps(rows,indent=2)+"\n")
    fields=list(rows[0]) if rows else ["model","task","documents_completed","documents_expected","complete"]
    with (out/"comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    teacher={f"{s['model']}|{s['task']}":s["teacher_mode"] for s in local}
    (out/"teacher_mode_test.json").write_text(json.dumps(teacher,indent=2)+"\n")
    lines=["# Exhaustive benchmark results", "", "**Status: active/resumable. Only actual test-split aggregate measurements are shown below. Partial cohorts are explicitly marked and are not represented as full-cohort scores.**", "", "## Old interface versus exhaustive interface", "", "All values use the same task test cohort and primary metric. For span-label tasks, precision/recall/F1 are pooled micro scores; coreference uses official CoNLL F1; speakers use joint quote/speaker B³. `Δ` is exhaustive minus the comparable score. The BookNLP score is a separate local pipeline run on the same documents. A row is complete only when completed documents equal the baseline cohort size.", "", "| Model | Task | Completed / expected | Exhaustive P / R / F1 | Old method F1 | Δ vs old | BookNLP small F1 | Δ vs BookNLP | Calls | Wall h | Output tok/s | Peak GPU / RAM MiB | Invalid | Status |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        def fmt(v): return "—" if v is None else f"{v:.4f}"
        prf=f"{fmt(r['exhaustive_precision'])} / {fmt(r['exhaustive_recall'])} / {fmt(r['exhaustive_f1_or_primary'])}"
        gpu="—" if r["peak_gpu_used_mib"] is None else str(r["peak_gpu_used_mib"])
        ram="—" if r["peak_system_ram_used_mib"] is None else str(r["peak_system_ram_used_mib"])
        tps="—" if r["output_tokens_per_second"] is None else f"{r['output_tokens_per_second']:.2f}"
        lines.append(f"| {r['model']} | {r['task']} | {r['documents_completed']}/{r['documents_expected']} | {prf} | {fmt(r['old_method_f1_or_primary'])} | {fmt(r['delta_vs_old'])} | {fmt(r['booknlp_small_f1_or_primary'])} | {fmt(r['delta_vs_booknlp'])} | {r['inference_calls']} | {r['wall_seconds']/3600:.2f} | {tps} | {gpu} / {ram} | {r['invalid_batches']+r['invalid_followups']} | {r['status']} |")
    if not rows: lines.extend(["| — | — | 0/— | — | — | — | — | — | — | — | — | — / — | — | pending |"])
    ablations=[]
    for s in local:
        if not s.get("ablation_metrics"): continue
        stages=[]
        for stage,values in s["ablation_metrics"].items():
            score=values.get("pooled_micro",{}).get("f1")
            if score is None: score=values.get("f1")
            if score is None:
                m=values.get("metrics",{})
                score=m.get("f1",m.get("CoNLL_F1",m.get("macro_joint_speaker_B3_F1")))
            if score is not None: stages.append((stage,score))
        if stages: ablations.append((s["model"],s["task"],s["documents"],stages))
    lines += ["", "## Inference-stage ablations", "", "Stages are measured on the same completed cohort; these are descriptive ablations, not independently tuned test-set prompts.", ""]
    if ablations:
        lines += ["| Model | Task | Documents | Stage F1 (stage: score) |", "| --- | --- | ---: | --- |"]
        for model,task,n,stages in ablations:
            rendered="; ".join(f"{name}: {score:.4f}" for name,score in stages)
            lines.append(f"| {model} | {task} | {n} | {rendered} |")
    else: lines.append("No complete test-cohort ablation summaries yet.")
    lines += ["", "`results/comparison.csv` and `.json` contain the test values and resource fields. Per-model/task JSON summaries include per-document scores, pass-level ablations, teacher-mode curves, and inference counters. Private token-level generations remain in the local experiment folder and are not part of this export.", "", "## Development pilots (not held-out results)", ""]
    if validation:
        lines += ["These runs are development/smoke diagnostics only. Do not rank them against test rows or present them as generalization estimates.", "", "| Model | Task | Documents | Primary metric | Calls | Model hours | Invalid batches |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for s in validation:
            score=f1(s); inv=s['inference']['invalid_batches']
            lines.append(f"| {s['model']} | {s['task']} | {s['documents']} | {score:.4f} | {s['inference']['calls']} | {s['inference']['wall_seconds']/3600:.2f} | {inv} |")
        lines += ["", "Per-document validation metrics and ablation layers are in the local summaries. One document is not a stable quality estimate."]
    else: lines.append("No validation pilot summaries have been recorded.")
    lines += ["", "## Interpretation guardrails", "", "- `coref` uses official CoNLL F1; `speakers` uses joint quote/speaker B³ F1. Conditional speaker B³ is separately stored in each task summary.", "- The original benchmark has independent zero-shot whole-excerpt JSON generations. The exhaustive approach uses one-token questions, multiple views, and task-specific post-processing; an observed difference cannot be attributed to a single component.", "- Teacher-mode confidence uses agreement/model ratings and is not calibrated.", "- The LitBank annotation layers are about 2,000-token excerpts, not complete novels. No result establishes multilingual performance.", ""]
    (HERE/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")

if __name__=="__main__": main()
