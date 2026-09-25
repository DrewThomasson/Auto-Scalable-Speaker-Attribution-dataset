# Auto-Scalable Speaker Attribution Dataset

This repository explores scalable literary dialogue detection and speaker attribution. Its original prototypes, model experiments, and data-generation tools are preserved in their existing folders; an index to those earlier materials is in [`archive/legacy/README.md`](archive/legacy/README.md).

## New benchmark: local LLMs vs. BookNLP

The completed [BookNLP local LLM benchmark](benchmark-results/booknlp-local-llm-2026/README.md) evaluates three quantized local instruction models and the released BookNLP small pipeline against human annotations. It covers entity typing, literary events, coreference, quotation boundaries, quotation speakers, and supersenses. Each model receives an isolated task prompt and unannotated numbered tokens; model output is schema-checked JSON, and malformed output is retained as a failed prediction rather than omitted.

### Held-out test scores

| System | Entities F1 | Events F1 | Coreference CoNLL F1 | Quotes F1 | Joint speaker B³ F1 | Supersenses F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BookNLP small (local run) | **0.7409** | **0.7036** | **0.6604** | **0.7833** | **0.4216** | **0.7646** |
| Qwen3.5 4B, Q4_K_M | 0.1193 | 0.1658 | 0.0632 | 0.2765 | 0.1790 | 0.1741 |
| Qwen3.5 9B, Q4_K_M | 0.1258 | 0.0621 | 0.0858 | 0.4246 | 0.1418 | 0.1371 |
| Gemma 4 12B, Q4_K_M | 0.2342 | 0.1557 | 0.2435 | 0.6990 | 0.2996 | 0.2116 |
| DeepSeek V4.1 Flash (OpenRouter / DeepInfra) | 0.3446 | 0.3521 | 0.3099 | 0.7646 | 0.2585 | 0.5380 |
| GPT-6 Luna Pro (OpenRouter / OpenAI default) | 0.2866 | 0.4808 | 0.3751 | 0.7214 | **0.5719** | 0.6389 |

Scores are pooled micro F1 for entities, events, quote boundaries, and supersenses; official CoNLL F1 for coreference; and joint quote/speaker B³ macro F1 for speakers. Test sizes by task are 10 / 30 / 100 / 10 / 10 / 35 documents, respectively. Published BookNLP numbers use different protocols and are reported separately in the benchmark documentation.

**Current finding:** BookNLP small leads GPT-6 Luna Pro on five of six headline tasks; Luna Pro has the best joint speaker B³ on this cohort (0.5719 vs. BookNLP 0.4216). Luna Pro scores 0.7214 on quote-only F1 and 0.7627 conditional speaker B³. Its full 195-request run cost $1.870728935, with 194 schema-valid outputs. See the [complete GPT-6 Luna Pro protocol and results](benchmark-results/hosted-gpt-6-luna-pro/README.md) and [DeepSeek results](benchmark-results/hosted-deepseek-v4.1-flash/README.md). This is an English test and does not validate multilingual performance or downstream student quality.

### New experiment: exhaustive atomic inference (in progress)

The [exhaustive maximum-accuracy benchmark](benchmark-results/booknlp-local-llm-exhaustive-2026/README.md) tests a different inference strategy: software asks a separate local model question for every token, then adds independent wider-context votes, span verification, pairwise coreference decisions, and consistency constraints. It reuses the same frozen gold cohorts and scorers, adds Qwen3.5 0.8B, and keeps all original results above unchanged. Two complete held-out event cohorts are now measured:

| Model / method | P | R | F1 | Against prior method | BookNLP-small F1 | Calls | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5 0.8B exhaustive | 0.0395 | 0.6987 | 0.0748 | No 0.8B baseline | 0.7036 | 127,577 | 12.43 h |
| Qwen3.5 4B original prompt | 0.1406 | 0.2019 | 0.1658 | — | 0.7036 | — | — |
| Qwen3.5 4B exhaustive | 0.2773 | 0.5504 | 0.3688 | +0.2030 F1 | 0.7036 | 127,457 | 18.94 h |

Qwen 0.8B found many gold events but overpredicted heavily; its uncalibrated top-25%-coverage precision was 0.0365. Exhaustive Qwen 4B improved its original-prompt F1 by 0.2030 on the full 30-excerpt test set, including gains in precision and recall, but still trails BookNLP-small by 0.3348 F1. Its uncalibrated top-25%-coverage precision was 0.2840, not yet a high-precision teacher subset. The 4B ablation scored F1 0.3026 after pass 1, 0.3736 after two-pass voting, and 0.3688 after final disagreement adjudication. Qwen3.5 9B events is running; results for the remaining models/tasks are pending. See the [full status, comparisons, and ablations](benchmark-results/booknlp-local-llm-exhaustive-2026/RESULTS.md), [comparison CSV](benchmark-results/booknlp-local-llm-exhaustive-2026/results/comparison.csv), [0.8B cohort summary](benchmark-results/booknlp-local-llm-exhaustive-2026/results/summaries/qwen3.5_0.8b_test_events.json), and [4B cohort summary](benchmark-results/booknlp-local-llm-exhaustive-2026/results/summaries/qwen3.5_4b_test_events.json).

### Jev 1.13 event evaluation (hosted decision model)

This is the result of the recent Jev test. Jev is a hosted decision model, so we used its native yes/no-per-token API rather than asking it to generate JSON spans. It was evaluated against the same held-out 30-book event set and gold labels as BookNLP small.

| System | Event precision | Event recall | Event micro F1 | Documents | Token questions answered | Reported API cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BookNLP small (local run) | 0.7452 | 0.6664 | **0.7036** | 30 | — | — |
| Jev 1.13 (OpenRouter) | 0.1799 | 0.8041 | **0.2940** | 30 | 63,625 / 63,625 | $0.5524 |

For Jev, event predictions use the fixed probability threshold 0.5. The high recall and low precision mean it labels many non-event tokens as events; its F1 is below BookNLP small. This interface-adapted result covers event triggers only and is not a local-model result. See the [full Jev protocol and result details](benchmark-results/booknlp-local-llm-2026/JEV_EVENT_SUPPLEMENT.md), [exact prompt template](benchmark-results/booknlp-local-llm-2026/prompts/jev_events.md), and [machine-readable result](benchmark-results/booknlp-local-llm-2026/results/jev_test_events.json).

## Explore the benchmark

- [Benchmark landing page, methodology, limitations, and reproduction notes](benchmark-results/booknlp-local-llm-2026/README.md)
- [Detailed scores and coverage, including invalid-output counts](benchmark-results/booknlp-local-llm-2026/RESULTS.md)
- [Exact task prompt templates](benchmark-results/booknlp-local-llm-2026/prompts/)
- [Machine-readable test metrics (CSV and JSON)](benchmark-results/booknlp-local-llm-2026/results/)
- [Dataset split manifest and measured model/runtime inventory](benchmark-results/booknlp-local-llm-2026/metadata/)
- [GPT-6 Luna Pro hosted benchmark, scores, cost, and validation](benchmark-results/hosted-gpt-6-luna-pro/README.md)
- [DeepSeek V4.1 Flash hosted benchmark, scores, cost, and validation](benchmark-results/hosted-deepseek-v4.1-flash/README.md)
- [Previous GPT-4/manual review and model-training experiments](archive/legacy/README.md)

The benchmark export contains aggregate metrics and prompt/configuration metadata. It does not redistribute LitBank/SemCor source data, book text, human gold annotations, or per-document model generations. Those licensed evaluation assets and detailed local artifacts are kept in the separate local experiment folder.

## Original project

The original GUI and scripts remain available: start with [`run_gui.py`](run_gui.py), [`speaker_find_attribute.py`](speaker_find_attribute.py), and [`manual_results_checker.py`](manual_results_checker.py). The earlier one-book GPT-4 review is an informal historical experiment, not a score directly comparable to the held-out benchmark.

Never commit API keys or credentials.
