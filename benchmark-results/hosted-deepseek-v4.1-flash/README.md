# DeepSeek V4.1 Flash on the frozen BookNLP-style benchmark

**Completed 23 September 2026 · primary benchmark · 195 one-document requests**

This is a hosted-model extension of the existing [local LLM benchmark](../booknlp-local-llm-2026/README.md). It answers how `deepseek/deepseek-v4.1-flash` performs under the same six-task, zero-shot, whole-document interface used for Qwen and Gemma. It does not use the separate exhaustive/atomic inference approach.

## Results

| System | Entities F1 | Events F1 | Coref CoNLL F1 | Quotes F1 | Joint speaker B³ F1 | Supersenses F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BookNLP small, local | 0.7409 | 0.7036 | 0.6604 | **0.7833** | **0.4216** | 0.7646 |
| Qwen3.5 4B, Q4_K_M | 0.1193 | 0.1658 | 0.0632 | 0.2765 | 0.1790 | 0.1741 |
| Qwen3.5 9B, Q4_K_M | 0.1258 | 0.0621 | 0.0858 | 0.4246 | 0.1418 | 0.1371 |
| Gemma 4 12B, Q4_K_M | 0.2342 | 0.1557 | 0.2435 | 0.6990 | **0.2996** | 0.2116 |
| **DeepSeek V4.1 Flash (DeepInfra via OpenRouter)** | **0.3446** | **0.3521** | **0.3099** | **0.7646** | **0.2585** | **0.5380** |

Entities, events, quotes, and supersenses are pooled micro F1. Coreference is official pooled CoNLL F1 (mean of MUC, B³, and CEAF-e F1). Speaker is the original macro joint quote/speaker B³ score. Cohort sizes are 10 / 30 / 100 / 10 / 10 / 35 documents. Old results are unchanged. Detailed precision/recall, speaker scores, output validity, and per-task usage are in [the full comparison](../booknlp-local-llm-2026/RESULTS.md) and the [machine-readable results](results/deepseek_test.csv).

### Quote and speaker details

| Measure | DeepSeek | Qwen 4B | Qwen 9B | Gemma 12B | BookNLP small |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quote task pooled precision | 0.7665 | 0.3310 | 0.4301 | 0.6729 | 0.8108 |
| Quote task pooled recall | 0.7626 | 0.2374 | 0.4192 | 0.7273 | 0.7576 |
| Quote task pooled F1 | 0.7646 | 0.2765 | 0.4246 | 0.6990 | 0.7833 |
| Speaker task quote-boundary macro F1 | 0.3097 | 0.2079 | 0.1947 | 0.3722 | 0.5501 |
| Conditional speaker B³ F1, exact quote matches only | 0.5091 | 0.5247 | 0.5332 | 0.5565 | 0.5412 |
| Joint quote/speaker B³ F1 | 0.2585 | 0.1790 | 0.1418 | 0.2996 | 0.4216 |

DeepSeek has the best quote-task F1 among the tested LLMs, within 0.0188 of BookNLP on this cohort. When the quote is exactly found, its speaker B³ is 0.5091; its joint score is lower because the separate speaker task's quote-boundary recall is only 0.2932. This distinction matters: quote detection is comparatively strong, while end-to-end speaker labeling is not the best tested LLM result.

## Protocol and reproducibility

- **Model:** exact OpenRouter slug `deepseek/deepseek-v4.1-flash`; no model substitution.
- **Provider:** pinned to DeepInfra with provider fallback disabled. All 195 responses reported DeepInfra and the exact requested model slug.
- **Endpoint:** `https://openrouter.ai/api/v1/chat/completions`.
- **Prompts and data:** the byte-identical existing task prompt files and frozen test IDs were used: entities v2, events v1, coref v2, quotes v1, speakers v1, and supersenses v1. SHA-256 hashes are recorded in `metadata/run.json`. The existing `benchmark.prompt_for` serialization appended every token as `[zero_based_id] token`, one full document and one task per request. No gold, BookNLP prediction, score, example, retrieval, consensus, or iterative correction was sent.
- **Generation:** temperature 0; `max_tokens=16384`; strict JSON Schema using the original task schema; `reasoning.effort="none"` and `include_reasoning=false`. The endpoint advertised these settings and strict structured outputs. The model reported zero reasoning tokens in the preflight request.
- **Context:** the local runs set Ollama `num_ctx=32768`; OpenRouter exposes no equivalent per-request context cap. We sent the same complete, unchunked documents and did not use the hosted 1,048,576-token context to combine documents. The selected endpoint advertised 1,048,576 context and 131,072 maximum completion tokens.
- **Output failures:** invalid or truncated model answers were scored as empty predictions under the existing parser policy and were never regenerated. Only transport/provider failures are eligible for retry. The 195 primary requests all returned HTTP 200 on their first attempt.
- **Caching:** per-request cache identity is derived from model, benchmark/split, task/document, exact serialized prompt, schema, and inference settings. Responses are atomically cached outside this repository and resumed without repeat requests. Raw generations and licensed test text are not included in this export.
- **Scoring:** the existing parser and metrics are imported from the local experiment package. Events are pooled across document-local token IDs; coreference is recomputed with the official reference scorer v8.01; speaker scoring retains the existing conditional and joint B³ metrics.

The coreference test cohort is the union of the ten official held-out folds. As in the existing benchmark, this union includes the ten documents that appear in fold 0's development IDs, because those documents are held out in another fold. The independent audit confirms that this is the only validation/test ID overlap (10 coreference IDs); the other five tasks have none. No development labels, smoke outputs, or evaluation feedback were included in any test request. The separate smoke test used one entities validation document only.

The model and provider capability/pricing snapshot is preserved in [`metadata/openrouter_model.json`](metadata/openrouter_model.json), with the complete safe run configuration, prompt versions/hashes, software versions, and usage totals in [`metadata/run.json`](metadata/run.json). OpenRouter's [model page](https://openrouter.ai/deepseek/deepseek-v4.1-flash) and [structured outputs documentation](https://openrouter.ai/docs/guides/features/structured-outputs) describe the hosted model listing and JSON Schema feature. Metadata was checked at inference time; the saved snapshot is authoritative for this run.

## Usage and validity

| Task | Requests | Schema valid | Input tokens | Output tokens | Cached input tokens | API cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Entities | 10 | 10/10 | 116,603 | 7,626 | 0 | $0.01952734 |
| Events | 30 | 29/30 | 352,543 | 28,985 | 0 | $0.06152972 |
| Coreference | 100 | 100/100 | 1,183,094 | 260,723 | 25,088 | $0.27172987 |
| Quotes | 10 | 10/10 | 116,960 | 2,229 | 0 | $0.01731058 |
| Speakers | 10 | 10/10 | 117,270 | 6,136 | 0 | $0.01899492 |
| Supersenses | 35 | 29/35 | 460,979 | 431,487 | 8,448 | $0.24461436 |
| **Total** | **195** | **188/195** | **2,347,449** | **737,186** | **33,536** | **$0.63370679** |

OpenRouter's selected DeepInfra endpoint snapshot listed $0.14 per million uncached input tokens, $0.42 per million output tokens, and $0.0042 per million cached input tokens. The reported total above is the actual usage cost returned by the API. A separate one-document validation-split smoke request cost $0.00222418; it is excluded from all test scores. Combined smoke plus test API-reported cost was $0.63593097.

Seven outputs (one event, six supersense) reached the 16,384-token output cap and were not valid JSON. That is the only schema failure mode in this run. No request transport retries occurred. The longest/most failure-prone task was supersense enumeration: it consumed 431,487 output tokens and six of its 35 responses were invalid.

## Interpretation

Compared with the local models in the original comparison, DeepSeek is stronger on entities, events, coreference, quote boundaries, and supersenses. A later [GPT-6 Luna Pro benchmark](../hosted-gpt-6-luna-pro/README.md) adds another hosted model: Luna Pro exceeds DeepSeek on events, coreference, joint and conditional speaker attribution, and supersenses, while DeepSeek retains stronger entity and quote-only F1. DeepSeek does not beat BookNLP on any of the six headline scores. Gemma 12B has a higher joint speaker B³ (0.2996 versus 0.2585). DeepSeek's conditional speaker B³ (0.5091) is below BookNLP and all three local LLMs.

On recall, DeepSeek substantially improves over the local LLMs for entities (0.2338) and coreference (official B³ recall 0.1757), but entity recall remains low and coreference still misses many gold mentions/clusters. Event recall is 0.3461, quote-task recall 0.7626, and supersense recall 0.5452. Speaker-task quote-boundary recall is only 0.2932, which helps explain why good standalone quote detection does not translate into a comparable joint speaker score. The most troublesome output task was supersenses: six responses were truncated and it used 431,487 completion tokens; the weakest headline task relative to its baseline is joint speaker attribution.

The results make DeepSeek a plausible candidate for assisted quote annotation and prelabeling, especially given its 0.7646 quote F1 and 0.7626 recall. The later Luna Pro run has the stronger measured joint speaker result; neither benchmark supports unattended labeling without further validation: joint speaker F1 is 0.2585, below Gemma and BookNLP, and conditional speaker B³ is 0.5091. For a student-teacher pipeline, speaker labels would need filtering or human review. This English test does not establish multilingual performance or downstream student quality.

## Files

- [`results/deepseek_test.csv`](results/deepseek_test.csv): compact task metrics, P/R, validity, token usage, and cost.
- [`results/deepseek_test.json`](results/deepseek_test.json): safe aggregate scores and usage, without per-document predictions.
- [`results/validation_audit.json`](results/validation_audit.json): independent cohort, prompt identity, output replay, token-offset, aggregation, provider, and split checks.
- [`metadata/openrouter_model.json`](metadata/openrouter_model.json): model/provider capabilities, context, and rate snapshot.
- [`metadata/run.json`](metadata/run.json): model settings, split protocols, generation counts, and usage totals.
- `runner.py` and `validate_results.py`: resumable runner and independent validation implementation. They expect the original licensed experiment checkout at `/home/drew/booknlp_llm_experiment` (or a compatible `--experiment-root`) and read credentials at runtime; no key is stored here.

On the benchmark host, the completed run can be independently revalidated with:

```bash
cd /home/drew/booknlp_llm_experiment
PYTHONPATH=src .venv/bin/python external/Auto-Scalable-Speaker-Attribution-dataset/benchmark-results/hosted-deepseek-v4.1-flash/validate_results.py \
  --experiment-root /home/drew/booknlp_llm_experiment \
  --results-dir /home/drew/booknlp_llm_experiment/results/openrouter_deepseek_v4_1_flash/test
```

Rerunning `runner.py` with the same experiment root resumes from its per-request cache. This repository export cannot reproduce licensed inputs by itself; the matching private dataset/scoring checkout and runtime credentials are required.

The public export intentionally excludes API credentials, licensed book text, gold labels, and per-document generations.
