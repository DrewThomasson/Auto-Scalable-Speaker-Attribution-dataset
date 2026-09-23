# Exhaustive benchmark results

**Status: active/resumable. Only actual test-split aggregate measurements are shown below. Partial cohorts are explicitly marked and are not represented as full-cohort scores.**

## Old interface versus exhaustive interface

All values use the same task test cohort and metric. `Δ` is exhaustive minus old-interface score; the BookNLP score is a separate local pipeline run on the same documents. A row is complete only when completed documents equal the baseline cohort size.

| Model | Task | Completed / expected | Old method | Exhaustive | Δ vs old | BookNLP small | Δ vs BookNLP | Calls | Wall h | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| — | — | 0/— | — | — | — | — | — | — | — | pending |

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
