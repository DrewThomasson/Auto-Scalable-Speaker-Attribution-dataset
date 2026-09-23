# Legacy experiments and results

This archive index separates the repository's earlier exploratory results from the completed held-out local benchmark. These earlier tests use different books, labels, models, and/or evaluation procedures; their numbers must not be compared as if they shared the new benchmark's gold annotations or split.

## 2023 GPT-4 one-book quote review

The original README reported manual speaker-attribution accuracy of 98.33% and 96.67% on two GPT-4 runs over a snippet from *Guardians of Ga'Hoole*. It describes 60 quotes and human categories such as correct attribution, incorrect attribution, and potentially misidentified quotation. It did not use a shared public test split, exact-boundary scoring, or the speaker B³ metric used in the new benchmark. Treat these as historical, manually reviewed prototype results only. The original README is preserved at [`README-pre-local-benchmark.md`](README-pre-local-benchmark.md); its sample-related files remain in `../../ebooks/`.

## Prior model and pipeline experiments

- `../../deepseek-testing/` contains an early 1.5B distilled DeepSeek experiment with an illustrative generated story and quote/speaker files. It is not evaluated against the benchmark gold datasets.
- `../../pipeline/`, `../../quote_identifier_BERT/`, `../../simple_gpt-2/`, `../../test_BERT/`, and `../../test_big_bird/` contain earlier BERT, GPT-2, and BigBird data preparation and training prototypes, model outputs, sample BookNLP output, and exploratory logs. Their local accuracy/evaluation numbers (where present) use their own splits and labels.
- `../../ebooks/` contains the early speaker-attribution sample and manual review files. Do not interpret those files as the new held-out benchmark or redistribute source book text without checking its rights.

The older source folders have been left in their original locations to preserve scripts' relative paths. The top-level README now links the completed benchmark as the current, clearly separated evaluation; this directory is the archive for prior-result descriptions and the superseded README.

## Historical provenance

The prior README's prose is stored without edits in [`README-pre-local-benchmark.md`](README-pre-local-benchmark.md). Its historical accuracy claims are reproduced only to preserve provenance; they are not endorsed as statistically comparable benchmark scores.
