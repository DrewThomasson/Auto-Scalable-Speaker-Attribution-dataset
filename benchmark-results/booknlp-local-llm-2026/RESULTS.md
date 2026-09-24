# Results and detailed protocol

## Full test-set comparison

All local LLM rows below are full held-out cohorts; the DeepSeek row is the separate hosted extension using the same frozen cohorts. The earlier one-book Gemma entity pilot (F1 0.4286) is superseded by the full 10-document score (0.2342). Earlier 10-book Qwen coreference pilots are superseded by the 100-book results here.

| System | Entities F1 (n=10) | Events F1 (n=30) | Coref CoNLL F1 (n=100) | Quotes F1 (n=10) | Joint speaker B³ F1 (n=10) | Supersenses F1 (n=35) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BookNLP small, local run | 0.7409 | 0.7036 | 0.6604 | 0.7833 | 0.4216 | 0.7646 |
| Qwen3.5 4B, Q4_K_M | 0.1193 | 0.1658 | 0.0632 | 0.2765 | 0.1790 | 0.1741 |
| Qwen3.5 9B, Q4_K_M | 0.1258 | 0.0621 | 0.0858 | 0.4246 | 0.1418 | 0.1371 |
| Gemma 4 12B, Q4_K_M | 0.2342 | 0.1557 | 0.2435 | 0.6990 | 0.2996 | 0.2116 |
| DeepSeek V4.1 Flash, OpenRouter / DeepInfra | 0.3446 | 0.3521 | 0.3099 | 0.7646 | 0.2585 | 0.5380 |
| GPT-6 Luna Pro, OpenRouter / OpenAI default | 0.2866 | 0.4808 | 0.3751 | 0.7214 | **0.5719** | 0.6389 |

The count in each heading is the number of evaluated documents. Entities, events, quote boundaries, and supersenses use pooled micro P/R/F1; coreference uses official CoNLL F1; speakers use the joint quote-and-speaker B³ macro F1. Because the task datasets and split designs differ, do not average these six columns into a general “accuracy” score.

## Task coverage, P/R, and output validity

`Valid` is the number of responses that parsed as JSON and passed the task schema, out of the full expected document count. An invalid response contributes an empty prediction; it is not silently discarded.

| Task | System | Documents | Pooled/primary score (P / R) | Valid |
| --- | --- | ---: | --- | ---: |
| Entities | BookNLP small | 10/10 | F1 .7409 (.8028 / .6878) | n/a |
|  | Qwen 4B | 10/10 | F1 .1193 (.2419 / .0792) | 10/10 |
|  | Qwen 9B | 10/10 | F1 .1258 (.3446 / .0769) | 10/10 |
|  | Gemma 12B | 10/10 | F1 .2342 (.5063 / .1523) | 9/10 |
|  | DeepSeek V4.1 Flash | 10/10 | F1 .3446 (.6554 / .2338) | 10/10 |
|  | GPT-6 Luna Pro | 10/10 | F1 .2866 (.3528 / .2413) | 9/10 |
| Events | BookNLP small | 30/30 | F1 .7036 (.7452 / .6664) | n/a |
|  | Qwen 4B | 30/30 | F1 .1658 (.1406 / .2019) | 30/30 |
|  | Qwen 9B | 30/30 | F1 .0621 (.3379 / .0342) | 30/30 |
|  | Gemma 12B | 30/30 | F1 .1557 (.4182 / .0957) | 30/30 |
|  | DeepSeek V4.1 Flash | 30/30 | F1 .3521 (.3582 / .3461) | 29/30 |
|  | GPT-6 Luna Pro | 30/30 | F1 .4808 (.3806 / .6525) | 30/30 |
| Coref | BookNLP small | 100/100 | CoNLL F1 .6604 | n/a |
|  | Qwen 4B | 100/100 | CoNLL F1 .0632 | 98/100 |
|  | Qwen 9B | 100/100 | CoNLL F1 .0858 | 100/100 |
|  | Gemma 12B | 100/100 | CoNLL F1 .2435 | 94/100 |
|  | DeepSeek V4.1 Flash | 100/100 | CoNLL F1 .3099; MUC P/R/F1 .7314/.3567/.4795; B³ P/R/F1 .6024/.1757/.2721; CEAF-e P/R/F1 .4378/.1117/.1780 | 100/100 |
|  | GPT-6 Luna Pro | 100/100 | CoNLL F1 .3751; MUC .5651; B³ .3541; CEAF-e .2061 | 100/100 |
| Quotes | BookNLP small | 10/10 | F1 .7833 (.8108 / .7576) | n/a |
|  | Qwen 4B | 10/10 | F1 .2765 (.3310 / .2374) | 10/10 |
|  | Qwen 9B | 10/10 | F1 .4246 (.4301 / .4192) | 10/10 |
|  | Gemma 12B | 10/10 | F1 .6990 (.6729 / .7273) | 10/10 |
|  | DeepSeek V4.1 Flash | 10/10 | F1 .7646 (.7665 / .7626) | 10/10 |
|  | GPT-6 Luna Pro | 10/10 | F1 .7214 (.7108 / .7323) | 10/10 |
| Speakers | BookNLP small | 10/10 | Joint B³ .4216; conditional B³ .5412 | n/a |
|  | Qwen 4B | 10/10 | Joint B³ .1790; conditional B³ .5247 | 10/10 |
|  | Qwen 9B | 10/10 | Joint B³ .1418; conditional B³ .5332 | 10/10 |
|  | Gemma 12B | 10/10 | Joint B³ .2996; conditional B³ .5565 | 10/10 |
|  | DeepSeek V4.1 Flash | 10/10 | Joint B³ .2585; conditional B³ .5091; speaker-task quote P/R/F1 .3336/.2932/.3097 | 10/10 |
|  | GPT-6 Luna Pro | 10/10 | Joint B³ .5719; conditional B³ .7627; speaker-task quote P/R/F1 .5934/.6166/.6018 | 10/10 |
| Supersenses | BookNLP small | 35/35 | F1 .7646 (.6525 / .9231) | n/a |
|  | Qwen 4B | 35/35 | F1 .1741 (.3300 / .1182) | 32/35 |
|  | Qwen 9B | 35/35 | F1 .1371 (.2362 / .0966) | 29/35 |
|  | Gemma 12B | 35/35 | F1 .2116 (.4110 / .1424) | 28/35 |
|  | DeepSeek V4.1 Flash | 35/35 | F1 .5380 (.5310 / .5452) | 29/35 |
|  | GPT-6 Luna Pro | 35/35 | F1 .6389 (.5584 / .7465) | 35/35 |

The complete hosted protocols, model metadata, per-task cost and token usage, validation audits, and interpretation are available in the [GPT-6 Luna Pro report](../hosted-gpt-6-luna-pro/README.md) and [DeepSeek V4.1 Flash report](../hosted-deepseek-v4.1-flash/README.md). Both hosted runs use the same frozen cohorts and exact prompts/schemas as the local models. Luna Pro used strict JSON Schema, reasoning effort `none`, and a 16,384-token `max_tokens` request; temperature was omitted because its OpenAI endpoint did not advertise it. OpenRouter reports the default service tier. See its report for the resulting protocol differences and usage.

## Metric definitions

- **Entities:** pooled exact typed span precision, recall, and F1. Start, end, and entity type must all match. Nested gold mentions remain in scoring.
- **Events:** pooled binary positive-token precision, recall, and F1, following the public literary event evaluator. Predicted trigger spans expand to their token sets.
- **Coreference:** official CoNLL v8.01 Perl scorer average of MUC, B³, and CEAF-e F1. The components for BookNLP / Qwen 4B / Qwen 9B / Gemma 12B are respectively MUC `.8200/.1007/.1215/.4284`, B³ `.6084/.0410/.0531/.1948`, and CEAF-e `.5527/.0480/.0828/.1074`. The displayed CoNLL score is their arithmetic mean. A separate document-macro CoNLL aggregation is `.6499/.0575/.0807/.2319`; it is not the pooled score in the main table.
- **Quotes:** pooled exact quote-boundary span F1; both start and end token must match.
- **Speaker attribution:** B³ over speaker clusters. The main table uses a **joint** score that includes missed/incorrect quote boundaries. Conditional speaker B³ is calculated only on exactly matched quote spans and is shown in the detailed coverage table to expose attribution quality when the quote is found.
- **Supersenses:** pooled exact word-span plus supersense-label F1 across 41 WordNet lexnames.

The official CoNLL scorer aborts if a prediction contains too many duplicate mention spans. Its input was therefore normalized by deterministically collapsing repeated exact predicted spans and keeping the first cluster assignment. This affects only scorer input: raw predictions are unchanged. The number removed was 1 / 11 / 20 for Qwen 4B / Qwen 9B / Gemma; conflicting cluster labels among those duplicates were 1 / 2 / 13. The tie rule and counts are noted here to make this scorer-specific accommodation explicit. Luna Pro required the same normalization: 36 repeated spans removed, with 34 duplicate-span cluster conflicts resolved by keeping the first assignment.

## Published BookNLP results (reference, not reproduced)

The BookNLP repository README reports these model figures:

| Published row | Entity F1 | Supersense F1 | Event F1 | Coreference average F1 | Speaker B³ F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BookNLP small | 0.882 | 0.732 | 0.706 | 0.764 | 0.864 |
| BookNLP big | 0.900 | 0.762 | 0.741 | 0.790 | 0.899 |

These are **published figures transcribed from BookNLP's README**, not runs reproduced in this local experiment. That table does not specify the same test cohorts and complete per-task protocols used here. Separate task papers also use different protocols: literary-event work reports 0.739 F1 for its best model; the coreference paper reports 0.681 average B³/MUC/CEAF-φ4 in 10-fold cross-validation; the quotation paper reports 0.908 F1 for regex quote detection and evaluates attribution with gold quote boundaries. These values are contextual references, not apples-to-apples model comparisons.

## Speed and local compute

| Model | Mean throughput | Peak measured post-call GPU use | Mean document time by task |
| --- | ---: | ---: | --- |
| Qwen3.5 4B | 58–64 tokens/s | up to 5.8 GiB | entities 22s; events 14s; coref 38s; quotes 13s; speakers 17s; supersenses 96s |
| Qwen3.5 9B | 42–48 tokens/s | about 7.5 GiB | entities 26s; events 12s; coref 40s; quotes 17s; speakers 21s; supersenses 173s |
| Gemma 4 12B | 33–35 tokens/s | about 9.0 GiB | entities 88s; events 21s; coref 133s; quotes 28s; speakers 52s; supersenses 233s |
| BookNLP small | not a generative model | CUDA | 7.5–7.8s per document after startup |

Hardware: RTX 3060 (12 GiB VRAM), 32 GiB system RAM. Mean generation speed is based on the runtime's generated-token and evaluation-duration counters. Per-document wall time includes malformed/invalid generations. The measured local GPU snapshots and complete task values are in `results/results_test_only.csv`.

## Interpretation

1. BookNLP small scored higher than every tested LLM on five headline tasks; GPT-6 Luna Pro exceeded it on joint speaker B³ in these cohorts (0.5719 vs. 0.4216). Its LitBank training exposure means this is not a clean unseen-LitBank generalization comparison.
2. Luna Pro leads the tested systems on joint speaker B³ (0.5719) and conditional speaker B³ (0.7627). DeepSeek remains stronger than Luna Pro on entity and quote-only F1; Luna Pro leads on events, coreference, joint/conditional speaker attribution, and supersenses.
3. The 9B Qwen does not dominate the 4B Qwen: it gains on entity spans and quote boundaries but loses on event triggers, joint speaker score, and supersenses. The 4B model is fastest and uses least VRAM.
4. Structured-output failures are task-dependent: Qwen 4B has 2 invalid coreference and 3 invalid supersense outputs; Qwen 9B has 6 invalid supersense outputs; Gemma has 1 entity, 6 coreference, and 7 supersense invalid outputs; DeepSeek has 1 event and 6 supersense invalid outputs, all truncated at its completion cap; Luna Pro has 1 malformed entity JSON response.
5. DeepSeek V4.1 Flash was the strongest of the local LLMs and remains the strongest tested LLM on quote-only F1 (0.7646). The later GPT-6 Luna Pro run scores higher on events, coreference, joint and conditional speaker attribution, and supersenses, while DeepSeek remains higher on entities and quote-only F1. Luna Pro records the strongest joint speaker B³ (0.5719) and conditional B³ (0.7627) in this cohort. DeepSeek's six supersense and one event invalid outputs all hit its 16,384-token output cap. See the separate [DeepSeek report](../hosted-deepseek-v4.1-flash/README.md) and [Luna Pro report](../hosted-gpt-6-luna-pro/README.md).
6. No model was evaluated on a multilingual gold set or used to train a student. These measurements cannot support a claim that a model is ready to label millions of examples or teach new languages. GPT-6 Luna Pro is a plausible English quote/speaker prelabeling candidate based on its 0.5719 joint speaker B³, but this does not establish human-level annotation, unattended-label quality, multilingual transfer, or downstream student gains. Human review and quality control remain important.

The local models' very low recall on dense supersense labeling and coreference suggests that the long JSON span format was difficult for them. DeepSeek has higher supersense recall (0.5452), while Luna Pro reaches 0.7465 recall; the task caused six DeepSeek output truncations but none for Luna Pro. These findings apply to the frozen interface used here. A different output representation would require a fresh, separately frozen experiment.

## Failure analysis availability

Per-document predictions and contextual failure examples are kept only in the local experiment folder because they contain test-book text and model generations. No copyrighted excerpts or predictions are reproduced in this public export. The aggregate error counts above remain auditable from the local `reports/failures/` and per-document result records.
