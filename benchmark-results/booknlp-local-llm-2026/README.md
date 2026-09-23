# Local LLMs on BookNLP-style literary annotation

**Completed local benchmark · 23 September 2026**

This is an experimental comparison of local instruction models against human-created literary annotations, with the released BookNLP small pipeline run on the same task cohorts. It is intended to answer whether these local models are ready to supply synthetic labels for a future student model. It does not train that student and does not establish performance on new languages.

## Result at a glance

| System | Parameters | Quantization | Entities F1 | Events F1 | Coref CoNLL F1 | Quote F1 | Joint speaker B³ F1 | Supersense F1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BookNLP small, local run | — | released checkpoint | **0.7409** | **0.7036** | **0.6604** | **0.7833** | **0.4216** | **0.7646** |
| Qwen3.5 4B | 4.66B | Q4_K_M | 0.1193 | 0.1658 | 0.0632 | 0.2765 | 0.1790 | 0.1741 |
| Qwen3.5 9B | 9.65B | Q4_K_M | 0.1258 | 0.0621 | 0.0858 | 0.4246 | 0.1418 | 0.1371 |
| Gemma 4 12B | 11.91B | Q4_K_M | 0.2342 | 0.1557 | 0.2435 | 0.6990 | 0.2996 | 0.2116 |

Every row above is a measured result on the full test cohort for the corresponding task. Test cohort sizes in column order are **10, 30, 100, 10, 10, and 35 documents**. These are distinct gold datasets and thus have different test sizes. The score aggregation is pooled micro F1 for entities, events, quote boundaries, and supersenses; official CoNLL F1 for coreference; and joint quote/speaker B³ macro F1 for speakers.

The BookNLP local run is a released pretrained pipeline; it has training exposure to LitBank, so its LitBank scores are not an unseen-domain estimate. BookNLP's published `small`/`big` scores and separate paper results are recorded in [`RESULTS.md`](RESULTS.md) with their protocol caveats; they are not substituted for the local run or directly ranked against this common test evaluation.

## What went into a model request

Inference was zero-shot, one task per request, one full document per request. For each held-out document the program serialized the task instruction followed by every token in the document with a stable zero-based token ID, like this schematic example:

```text
Identify every human or non-human entity mention ...
Return only JSON matching {"entities":[...]}

PASSAGE TOKENS (the integers are the only valid token IDs):
[0] Elizabeth
[1] walked
[2] into
[3] the
[4] room
[5] .
```

The example sentence is illustrative, not text from a test book. In the real call, the passage was the complete flattened token sequence from that test document. The LLM got the original token strings and token IDs, not BRAT/CoNLL labels, gold spans, clusters, speakers, task answers, or evaluation feedback. No few-shot examples, retrieval, chain-of-thought request, or consensus pass was used. Documents fit in the 32,768-token context used here; no context-window chunking was applied.

The prompt files in [`prompts/`](prompts/) are the exact task instruction text used by the benchmark, including its version marker and JSON schema. The final section `PASSAGE TOKENS` was appended by the runner. In coreference prompt v2, the leading `+` characters visible in the checked-in prompt file were also present in the loaded instruction; the prompt file is preserved byte-for-byte so this can be inspected and reproduced. The largest runtime-counted input prompt was 26,758 tokens, below the 32,768 context setting; no document was chunked. Long generated outputs could still be cut off, and those invalid JSON responses remain counted as failures.

### Task outputs and annotation scope

All token offsets are zero-based and inclusive. Each task is prompted separately; the model is asked for only that task's top-level JSON array:

| Task | Prompt asks the model for | JSON top-level key | Gold annotation scope |
| --- | --- | --- | --- |
| Entities | Mention start/end token and entity type | `entities` | LitBank PER, FAC, GPE, LOC, VEH, ORG spans; nested mentions retained |
| Events | One-token event trigger spans | `events` | LitBank asserted realis event triggers |
| Coreference | Mention spans, type, and a consistent entity cluster ID | `mentions` | LitBank identity mentions/clusters, including pronouns and singletons |
| Quote detection | Quote start/end boundaries, including delimiters when present | `quotes` | LitBank direct quotation spans |
| Speakers | Quote boundary, speaker mention span if present, and speaker cluster ID | `speakers` | LitBank quotation-to-character links |
| Supersenses | Word span and one of 41 WordNet lexicographer labels | `supersenses` | SemCor 3.0 word/sense annotations mapped to WordNet 3.0 lexnames |

See each exact task instruction in `prompts/{entities,events,coref,quotes,speakers,supersenses}.md`. The structured-output request used the task's JSON schema through Ollama, and the returned text was separately parsed and checked. Invalid JSON, schema violations, and output truncation are retained as invalid rows and scored as empty predictions. Gold annotations are not relaxed or edited to accommodate model output.

## Models, runtime, and generation settings

| Model tag | Family | Reported parameters | Quantization | Runtime | 32K context peak used GPU memory | Mean generation throughput |
| --- | --- | ---: | --- | --- | ---: | ---: |
| `qwen3.5:4b` | Qwen3.5 | 4.66B | Q4_K_M | Ollama 0.34.3 | up to 5.8 GiB | 58–64 tokens/s |
| `qwen3.5:9b` | Qwen3.5 | 9.65B | Q4_K_M | Ollama 0.34.3 | about 7.5 GiB | 42–48 tokens/s |
| `gemma4:12b` | Gemma 4 | 11.91B | Q4_K_M | Ollama 0.34.3 | about 9.0 GiB | 33–35 tokens/s |

Hardware was an NVIDIA GeForce RTX 3060 with 12,288 MiB VRAM and 32 GiB system RAM. All candidate models ran sequentially at full GPU placement; no CPU offload was used. LLM calls used temperature 0, thinking/reasoning disabled, `num_ctx=32768`, and `num_predict=16384`. Model digests, exact values, average per-document wall time, schema validity, and task-level measurements are included in [`metadata/measured_models.json`](metadata/measured_models.json) and [`results/results_test_only.csv`](results/results_test_only.csv). The BookNLP pipeline is not a generative LLM and has no comparable token/s figure.

## Datasets, splits, and licenses

- **Entities, events, coreference, quotations, speakers:** human-annotated LitBank documents. Task splits use the source authors' released document IDs for entity and event evaluation, and the published 10-fold test-fold union for coreference. Quote and speaker test IDs use one frozen SHA-256 10% document holdout because no shared official test split was available in this setup.
- **Supersenses:** human-annotated SemCor 3.0 documents mapped using WordNet 3.0. The 35-doc test set was frozen by SHA-256 holdout.
- The exact selected document IDs and split protocols are in [`metadata/splits.json`](metadata/splits.json). The benchmark used 100 LitBank coreference books, 30 event books, 10 entity books, 10 quote books, 10 speaker books, and 35 SemCor documents.
- Upstream sources include [LitBank](https://github.com/dbamman/litbank), [NAACL 2019 literary entities](https://github.com/dbamman/NAACL2019-literary-entities), [ACL 2019 literary events](https://github.com/dbamman/ACL2019-literary-events), [LREC 2020 literary coreference](https://github.com/dbamman/lrec2020-coref), [SemCor](https://web.eecs.umich.edu/~mihalcea/downloads.html#semcor), and [Princeton WordNet](https://wordnet.princeton.edu/). Check each upstream license and conditions before reusing source material.

This repository contains only split IDs, prompt templates, aggregate evaluation results, and methodology. It does not contain the underlying LitBank/SemCor documents, human gold labels, or per-document generations. The full local experiment and licensed source distributions remain outside this GitHub repository.

## Metrics and result files

The score meanings, per-task precision/recall, full coverage table, schema-valid counts, BookNLP published-reference figures, and interpretation are in [`RESULTS.md`](RESULTS.md). The compact test-only machine-readable exports are [`results/results_test_only.csv`](results/results_test_only.csv) and [`results/results_test_only.json`](results/results_test_only.json); all splits are separately available in the `summary_all_splits` files. Metrics follow the relevant task literature where available: pooled exact typed-span F1 for entities, pooled positive-token F1 for events, official CoNLL v8.01 MUC/B³/CEAF-e average for coreference, exact quote-boundary F1, speaker-cluster B³, and exact span+label F1 for supersenses.

## Interpretation and limitations

BookNLP small exceeds every tested local model on these full test cohorts. Gemma 4 12B is the strongest LLM for entities, coreference, quotes, and joint speaker attribution; it remains well behind BookNLP on those tasks. It also has the highest LLM supersense score, while Qwen 4B is the strongest Qwen model on events, speaker joint score, and supersenses. Qwen 9B is faster than Gemma and improves over Qwen 4B on entity and quote F1, but parameter count alone did not predict quality.

The aggregate evidence does not yet support using any tested model as an unattended, large-scale synthetic-data teacher. In particular, coreference F1 and output validity are weak, and this study contains no multilingual gold set or downstream student training. A future compact-label or multi-pass interface would be a new experiment, with its own frozen prompts and held-out evaluation.

## Supplementary Jev 1.13 event result

TypeSafe Jev 1.13 is a hosted decision model, not a local generative LLM. It was run separately through OpenRouter's native yes/no-per-token Decisions API on the same held-out 30-book event test split. With a fixed `P(yes) >= 0.5` threshold, Jev scored micro P/R/F1 **0.1799 / 0.8041 / 0.2940**, versus BookNLP small's **0.7452 / 0.6664 / 0.7036**. All 63,625 token decisions were answered. The request protocol is different from the JSON-span generation rows above, and this run evaluates events only. See [full Jev results and limitations](JEV_EVENT_SUPPLEMENT.md), [exact decision prompt](prompts/jev_events.md), and [JSON results](results/jev_test_events.json).

## Reproduction and provenance

The local experiment driver and full caches are not published here because prompts were applied to licensed held-out text and the user requested a one-folder local experiment. In the original local directory, `./run_remaining_cohorts.sh` reuses completed rows and caches; results and configuration are summarized in `reports/comparison.md`, `results/summary.csv`, and `results/summary.json`. The dataset/repository commits and exact runtime details are recorded there under `logs/` and `models/`. The full local experiment folder may be deleted independently of this public aggregate report.

No GitHub repositories were modified during the original benchmark. This dedicated export contains aggregate results and documentation only; it does not include the local models, dataset documents, raw prompts with book text, or model predictions at document level.

The `summary_all_splits` exports also include validation records used for development and early pilots. They are supplied for completeness only; use `results_test_only` for the frozen test comparison above. In particular, a validation set with one document is not a stable estimate of generalization.
