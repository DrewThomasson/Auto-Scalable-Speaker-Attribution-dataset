# GPT-6 Luna Pro on the frozen BookNLP-style benchmark

**Completed 23 September 2026 (local time) · 195 held-out requests · OpenRouter**

This hosted run adds GPT-6 Luna Pro to the existing six-task benchmark. Each request uses the same frozen test document, original task prompt, token-number serialization, JSON schema, and scorer as the local-model runs and the DeepSeek extension. It uses one task and one complete document per request. No gold labels, BookNLP predictions, examples, scores, or feedback were sent to the model.

## Results

| System | Entities F1 | Events F1 | Coref CoNLL F1 | Quotes F1 | Joint speaker B³ F1 | Supersenses F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BookNLP small, local | **0.7409** | **0.7036** | **0.6604** | **0.7833** | 0.4216 | **0.7646** |
| Qwen3.5 4B | 0.1193 | 0.1658 | 0.0632 | 0.2765 | 0.1790 | 0.1741 |
| Qwen3.5 9B | 0.1258 | 0.0621 | 0.0858 | 0.4246 | 0.1418 | 0.1371 |
| Gemma 4 12B | 0.2342 | 0.1557 | 0.2435 | 0.6990 | 0.2996 | 0.2116 |
| DeepSeek V4.1 Flash | 0.3446 | 0.3521 | 0.3099 | **0.7646** | 0.2585 | 0.5380 |
| **GPT-6 Luna Pro (OpenAI via OpenRouter)** | 0.2866 | **0.4808** | **0.3751** | 0.7214 | **0.5719** | **0.6389** |

Entities, events, quotes, and supersenses use pooled micro F1; coreference is official pooled CoNLL F1; speaker attribution uses macro joint quote/speaker B³ F1. Each task uses its existing cohort (10 / 30 / 100 / 10 / 10 / 35 documents). The prior rows and results are unchanged.

### Precision, recall, and speaker attribution

| Task / measure | Luna Pro | DeepSeek Flash | BookNLP small |
| --- | ---: | ---: | ---: |
| Entities P / R / F1 | .3528 / .2413 / .2866 | .6554 / .2338 / .3446 | .8028 / .6878 / .7409 |
| Events P / R / F1 | .3806 / .6525 / .4808 | .3582 / .3461 / .3521 | .7452 / .6664 / .7036 |
| Coreference CoNLL F1 | .3751 | .3099 | .6604 |
| Quotes P / R / F1 | .7108 / .7323 / .7214 | .7665 / .7626 / .7646 | .8108 / .7576 / .7833 |
| Speaker-task quote boundary P / R / F1 | .5934 / .6166 / .6018 | .3336 / .2932 / .3097 | .5501 (F1) |
| Conditional speaker B³ P / R / F1, exact quote matches | .7654 / .7600 / **.7627** | — / — / .5091 | — / — / .5412 |
| Joint quote/speaker B³ F1 | **.5719** | .2585 | .4216 |
| Supersenses P / R / F1 | .5584 / .7465 / .6389 | .5310 / .5452 / .5380 | .6525 / .9231 / .7646 |

The conditional speaker score evaluates speaker clustering only on exactly matched quote boundaries. Joint B³ includes quote-boundary detection errors. Luna Pro is the first tested model to exceed BookNLP small on the joint speaker measure in this cohort. Its quote-only F1 and the other four headline scores remain below BookNLP. Relative to DeepSeek, Luna Pro scores higher on events, coreference, joint speaker B³, and supersenses, but lower on entities and quote-only F1.

### Output validity and spend by task

| Task | Requests | Schema-valid | Input tokens | Output tokens | Total tokens | Cached input tokens | API cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Entities | 10 | 9/10 | 423,812 | 38,503 | 462,315 | 117,241 | $0.05232006 |
| Events | 30 | 30/30 | 1,252,671 | 89,092 | 1,341,763 | 353,783 | $0.14240278 |
| Coreference | 100 | 100/100 | 5,087,222 | 1,180,212 | 6,267,434 | 1,166,198 | $1.010598005 |
| Quotes | 10 | 10/10 | 390,733 | 4,912 | 395,645 | 117,271 | $0.032444785 |
| Speakers | 10 | 10/10 | 400,500 | 13,119 | 413,619 | 117,931 | $0.037749035 |
| Supersenses | 35 | 35/35 | 2,323,723 | 799,855 | 3,123,578 | 467,642 | $0.59521427 |
| **Total** | **195** | **194/195** | **9,878,661** | **2,125,693** | **12,004,354** | **2,340,066** | **$1.870728935** |

OpenRouter's generation metadata confirms 195 successful generation records, 195 OpenAI/default-tier routes, and the dated resolved model `openai/gpt-6-luna-pro-20260922`. The saved OpenRouter endpoint metadata lists default-route rates of $0.10 per million input tokens, $0.50 per million output tokens, $0.01 per million cached input tokens, and $0.125 per million cached input write tokens. Its independently summed cost and native token usage match the runner's response usage exactly. No request was retried. One entities response ("The Circular Staircase") was not parseable JSON; it was scored as an empty prediction and not regenerated. The one-document validation smoke request cost $0.00504692 and is excluded from the test scores and totals; smoke plus test cost was $1.875775855.

## Protocol and reproducibility

- **Requested model:** `openai/gpt-6-luna-pro`, with no substitution ([OpenRouter model page](https://openrouter.ai/openai/gpt-6-luna-pro)).
- **Provider and tier:** OpenAI, pinned by provider order with fallback disabled. No service tier was explicitly requested; OpenRouter generation records show `default` on all 195 requests. The resolved dated model is recorded above.
- **Endpoint:** `https://openrouter.ai/api/v1/chat/completions`.
- **Prompts and data:** byte-identical benchmark prompt templates and the existing frozen test IDs, with the exact `benchmark.prompt_for` serialization of every `[zero_based_id] token`. Cohort IDs and prompt hashes are in the metadata and audit. No chunking, examples, retrieval, consensus, or iterative correction.
- **Structured output:** strict OpenAI-compatible JSON Schema using the original task schema; OpenRouter endpoint metadata advertised structured outputs.
- **Reasoning:** the request set `reasoning.effort="none"` and `include_reasoning=false`; OpenRouter reported 4 reasoning tokens in total across the 195 responses. Luna Pro is nevertheless the separately named Pro serving variant, described as the underlying Luna model served in Pro mode.
- **Temperature:** omitted because the selected OpenAI endpoint did not advertise a temperature parameter. OpenRouter applied the provider default. This differs from the local `temperature=0` setting and is a limitation of the apples-to-apples match.
- **Context and output:** complete documents were sent in a single request. The baseline set Ollama `num_ctx=32768`; OpenRouter has no matching per-request setting. The default OpenAI endpoint advertised a 1,050,000-token context. Requests set `max_tokens=16384`; API-reported completion usage exceeded that value for 37 responses (maximum 29,346 tokens), while each had `finish_reason="stop"`. We retain the configured value and API-reported usage without reinterpretation.
- **Failure policy:** a received invalid answer is final and scored under the existing parser as an empty/cleaned prediction. Only transport/provider failures may be retried. There were zero transport failures or retries.
- **Resumability:** deterministic request hashes cover the requested model, benchmark/split, task/document, prompt, schema, and settings. Completed responses are atomically cached outside this repository. Re-running skips completed requests.
- **Scoring:** existing task parser and metrics from the private local experiment checkout; coreference uses the reference CoNLL v8.01 scorer. Independent validation replayed all saved outputs and recomputed per-task scores.

The run used the frozen 10 / 30 / 100 / 10 / 10 / 35 cohorts and scored each document independently, so token offsets never collide across documents. For the official coreference scorer, the existing deterministic accommodation removed 36 repeated mention spans and kept the first assignment for 34 duplicate-span cluster conflicts; raw model predictions were unchanged. Coreference test is the union of the ten held-out folds; as in the baseline, its ten fold-0 development IDs also appear in that pooled test union because they are held out in another fold. The independent audit checked this split-specific overlap and found no other validation/test overlap.

Regression tests are model-free and can be run with `python -m unittest discover -s benchmark-results/hosted-gpt-6-luna-pro/tests -v`. On the benchmark host, rerun the independent result audit with:

```bash
python benchmark-results/hosted-gpt-6-luna-pro/validate_results.py \
  --experiment-root /home/drew/booknlp_llm_experiment \
  --results-dir /home/drew/booknlp_llm_experiment/results/openrouter_gpt_6_luna_pro/test
```

## Interpretation

GPT-6 Luna Pro's standout result is quote/speaker attribution: joint B³ F1 is 0.5719, exceeding BookNLP small's 0.4216 on this shared test cohort; conditional speaker B³ is 0.7627. This makes Luna Pro a plausible English quote/speaker prelabeling teacher candidate for further evaluation, with the strongest measured joint speaker score among these systems. It does not show that it is human-level or that labels can be used unattended. Quote-only F1 is 0.7214, and Luna Pro remains below BookNLP on entities, events, coreference, quote boundaries, and supersenses. This is one English benchmark; multilingual transfer and downstream student quality remain untested.

## Files

- [`results/luna_test.csv`](results/luna_test.csv) and [`results/luna_test.json`](results/luna_test.json): aggregate per-task metrics, validity, and usage; no per-document generations.
- [`results/validation_audit.json`](results/validation_audit.json): independent split, prompt, identity, parsing, scoring, totals, and generation-metadata audit.
- [`metadata/openrouter_model.json`](metadata/openrouter_model.json): model listing and default OpenAI endpoint metadata snapshot.
- [`metadata/generation_usage.json`](metadata/generation_usage.json): aggregate OpenRouter generation metadata, cost, tier, routing, and token cross-checks; no generation IDs.
- [`metadata/run.json`](metadata/run.json): benchmark config, exact prompt versions/hashes, split protocol, totals, software version, and protocol differences.
- [`runner.py`](runner.py), [`validate_results.py`](validate_results.py), [`tests/test_runner.py`](tests/test_runner.py): resumable runner, independent validator, and OpenRouter-specific regression tests.

The public export excludes API credentials, licensed book text, gold labels, raw per-document outputs, and generation IDs. Reproduction requires the licensed local experiment checkout at `/home/drew/booknlp_llm_experiment` (or a compatible `--experiment-root`) and a runtime OpenRouter credential. The API key is read only at runtime and is never stored in this repository.
