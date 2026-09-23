# Supplementary Jev 1.13 event-detection result

**Run date:** 2026-09-23 · **OpenRouter model snapshot:** `typesafe/jev-1.13-20260917` · **Coverage:** 30/30 held-out LitBank event documents

## Same event test labels, Jev's native decision interface

Jev does not generate arbitrary text or span lists. OpenRouter exposes it through a typed Decisions API that returns yes/no probabilities. This supplementary experiment asked one binary event-trigger question for every token position and converted `P(yes) >= 0.5` to a one-token event span. The threshold was fixed before the test run.

| System | Interface | Precision | Recall | Micro F1 | Gold tokens | Predicted tokens | Matched |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BookNLP small, local run | BookNLP event sequence model | 0.7452 | 0.6664 | 0.7036 | 2,164 | 1,935 | 1,442 |
| Qwen3.5 4B Q4_K_M | Zero-shot JSON span generation | 0.1406 | 0.2019 | 0.1658 | 2,164 | 3,108 | 437 |
| Qwen3.5 9B Q4_K_M | Zero-shot JSON span generation | 0.3379 | 0.0342 | 0.0621 | 2,164 | 219 | 74 |
| Gemma 4 12B Q4_K_M | Zero-shot JSON span generation | 0.4182 | 0.0957 | 0.1557 | 2,164 | 495 | 207 |
| Jev 1.13 via OpenRouter | One yes/no decision per token | 0.1799 | 0.8041 | **0.2940** | 2,164 | 9,672 | 1,740 |

Jev answered all **63,625/63,625** candidate-token questions, with no failed batches. Its high recall came with low precision: it predicted over four times as many event tokens as the gold set. Its F1 is below BookNLP and above the three generative LLMs in this event task. This result covers events only and is not a JSON-generation comparison.

## Exactly what Jev received

Each request's `state` contained the complete document as numbered tokens (`[0] word`, `[1] word`, and so on). Its `questions` field contained up to 128 separate Noul questions, each naming one token ID and surface form and asking whether it was an asserted realis event trigger. The event criteria asked the model to exclude hypothetical, intended, future, counterfactual, and nonspecific narrated events. Each document was sent in full; no text windows or gold-derived candidate mentions were used. Human annotations never appeared in any request.

The [exact prompt description and templated question](prompts/jev_events.md) show the request construction without redistributing book text. The model returns a probability for each decision; the evaluator turns probabilities at or above 0.5 into positive trigger tokens. The full text was sent to the third-party OpenRouter/TypeSafe API as required to use this hosted model.

## API cost and throughput

- 510 Decisions API calls.
- 13,152,772 input tokens; reported usage cost **$0.552416**.
- Mean API latency: 0.529 seconds per call; summed request latency: 269.8 seconds.
- Largest runtime-counted input request: 30,920 tokens. All requests completed and returned every requested answer.
- Jev is a hosted proprietary model; local VRAM and tokens/second do not apply. The Decisions API does not expose temperature or reasoning controls.
- Before the held-out test, a one-document validation smoke run checked the multi-question request format (F1 .3312 on that single document; not a quality estimate) and cost $0.016860. Total usage billed across smoke plus test was **$0.569276**. Its [machine-readable smoke result](results/jev_validation_smoke.json) is separate from the test row.

The [OpenRouter Jev documentation](https://openrouter.ai/blog/insights/what-is-jev/) documents its typed decision output and endpoint. OpenRouter's [Jev model page](https://openrouter.ai/typesafe/jev-1.13) currently lists a 32K context and input pricing; the run's returned usage cost above is authoritative for this measured run.

## Scoring correction

The first internal score preview, 0.7280, was invalid. Its aggregation unioned token-index sets across books even though token IDs restart at zero in each document. A check against the benchmark's known 2,164 gold positive-token count exposed the collision. The corrected pooled calculation sums each document's matched, predicted, and gold counts before computing precision/recall/F1, yielding **0.2940**. The invalid preview was withdrawn. A regression test now covers pooling across documents with restarted offsets.

## Artifacts and rerun

- [Machine-readable aggregate result](results/jev_test_events.json)
- [CSV score and usage summary](results/jev_test_events.csv)
- [Exact question template](prompts/jev_events.md)
- [Split ID manifest](metadata/splits.json)
- In the local experiment, rerun from its root with `PYTHONPATH=src .venv/bin/python scripts/evaluate_jev_events.py --split test --batch-size 128`. Decisions are cached by request, so completed calls are reused. The script reads the OpenRouter key from the user's private config path and never includes it in result files.

Jev was not evaluated on entity spans, coreference clusters, quote boundaries, speaker attribution, or supersense tagging in this experiment. The number does not establish multilingual teacher suitability or downstream student performance.
