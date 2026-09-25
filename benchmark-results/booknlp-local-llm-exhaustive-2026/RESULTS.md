# Exhaustive benchmark results

**Status: active/resumable. Only actual test-split aggregate measurements are shown below. Partial cohorts are explicitly marked and are not represented as full-cohort scores.**

## Old interface versus exhaustive interface

All values use the same task test cohort and primary metric. For span-label tasks, precision/recall/F1 are pooled micro scores; coreference uses official CoNLL F1; speakers use joint quote/speaker B³. `Δ` is exhaustive minus the comparable score. The BookNLP score is a separate local pipeline run on the same documents. A row is complete only when completed documents equal the baseline cohort size.

| Model | Task | Completed / expected | Exhaustive P / R / F1 | Old method F1 | Δ vs old | BookNLP small F1 | Δ vs BookNLP | Calls | Wall h | Output tok/s | Peak GPU / RAM MiB | Invalid | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qwen3.5:0.8b | events | 30/30 | 0.0395 / 0.6987 / 0.0748 | — | — | 0.7036 | -0.6287 | 127577 | 12.43 | 25.72 | 1709 / 17219 | 0 | complete |
| qwen3.5:4b | events | 30/30 | 0.2773 / 0.5504 / 0.3688 | 0.1658 | 0.2030 | 0.7036 | -0.3348 | 127457 | 18.94 | 12.34 | 4287 / 15073 | 0 | complete |

## Inference-stage ablations

Stages are measured on the same completed cohort; these are descriptive ablations, not independently tuned test-set prompts.

| Model | Task | Documents | Stage F1 (stage: score) |
| --- | --- | ---: | --- |
| qwen3.5:0.8b | events | 30 | pass1: 0.0767; two_pass_vote: 0.0780; final: 0.0748 |
| qwen3.5:4b | events | 30 | pass1: 0.3026; two_pass_vote: 0.3736; final: 0.3688 |

`results/comparison.csv` and `.json` contain the test values and resource fields. Per-model/task JSON summaries include per-document scores, pass-level ablations, teacher-mode curves, and inference counters. Private token-level generations remain in the local experiment folder and are not part of this export.

## Development pilots (not held-out results)

These runs are development/smoke diagnostics only. Do not rank them against test rows or present them as generalization estimates.

| Model | Task | Documents | Primary metric | Calls | Model hours | Invalid batches |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| qwen3.5:0.8b | events | 1 | 0.0731 | 4011 | 0.39 | 0 |

Per-document validation metrics and ablation layers are in the local summaries. One document is not a stable quality estimate.

## Interpretation guardrails

- `coref` uses official CoNLL F1; `speakers` uses joint quote/speaker B³ F1. Conditional speaker B³ is separately stored in each task summary.
- The original benchmark has independent zero-shot whole-excerpt JSON generations. The exhaustive approach uses one-token questions, multiple views, and task-specific post-processing; an observed difference cannot be attributed to a single component.
- Teacher-mode confidence uses agreement/model ratings and is not calibrated.
- The LitBank annotation layers are about 2,000-token excerpts, not complete novels. No result establishes multilingual performance.
