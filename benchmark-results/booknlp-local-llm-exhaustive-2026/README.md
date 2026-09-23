# Exhaustive inference-time-compute benchmark

**Status: implementation and local validation in progress; no complete test-cohort exhaustive result is claimed yet.** This benchmark is designed to test whether low scores in the existing [document-enumeration benchmark](../booknlp-local-llm-2026/README.md) are partly caused by requiring one large annotation enumeration. It preserves that benchmark unchanged and runs beside it.

## Research design

The software, rather than the model, enumerates every token ID. A task-specific binary/multiclass decision is collected for each token in compact batches, with two independent judgments (reverse traversal and wider context on the second pass) and a third wider-context vote only where those judgments disagree. Spans are expanded separately for entity/coreference starts. Coreference uses pairwise identity questions and deterministic graph clustering. Quote boundaries are classified at every token and paired in reading order. Speaker questions operate over quotes proposed by this run's quote detector, never gold quotes. Supersenses are classified at every token from the benchmark's 41 WordNet 3.0 noun/verb labels.

The initial pass is intended as an atomic-enumeration ablation; the two-pass vote and final disagreement-resolved outputs are retained per token. A comparison of those layers isolates effects of a second judgment and disagreement handling. Confidence is vote agreement plus model-rated local span confidence where relevant; it is uncalibrated and must not be interpreted as a probability. Teacher-mode precision/coverage must be calculated on a development set before selecting thresholds; held-out coverage curves are descriptive only.

## Models and runtime

The three comparison weights are the same model tags, quantization, and Ollama local backend as the original benchmark: Qwen3.5 4B Q4_K_M (4.66B parameters), Qwen3.5 9B Q4_K_M (9.65B), and Gemma 4 12B Q4_K_M (11.91B). New size point: Qwen3.5 0.8B (Ollama reports 873.44M parameters, Q8_0, Apache-2.0), matching the Qwen family. The upstream [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-0.8B) and [Ollama package](https://ollama.com/library/qwen3.5:0.8b) identify the model/license/runtime artifact. The local tag digest, full blob SHA-256, and settings are in [`configs/models.json`](configs/models.json). Models and inference are local; Jev/OpenRouter is not used here.

Hardware: RTX 3060 with 12,288 MiB VRAM, 32 GiB RAM. Runtime is the experiment's Ollama 0.34.3 installation, with `HOME` and `OLLAMA_MODELS` redirected under `/home/drew/booknlp_llm_experiment`. Task context is 8,192, temperature 0, thinking off. Each token is a separate request (`num_predict=128` for one label); no weight updates or CPU-offloaded model substitution.

## Data and comparability

The runner calls the original benchmark's frozen split-selection logic and scorer. It uses the same LitBank task-layer normalized records and SemCor 3.0 35-document split as the original measured comparison. Important limitation from source inspection: the LitBank layers represent short annotated excerpts (mostly about 2,000 tokens), not complete raw novels. All test input contains passage tokens only; gold annotations are accessed after prediction by the scorer. The split manifest is [`../booknlp-local-llm-2026/metadata/splits.json`](../booknlp-local-llm-2026/metadata/splits.json). The local gold/text assets remain in the user's ignored experiment folder and are not copied into this repository.

The original scores remain the historical one-document/large-JSON benchmark. This experiment changes the inference interface and adds additional test-time compute, so direct deltas are protocol comparisons, not evidence that only one factor caused the difference. Both use the same frozen documents, model digests where available, and established scoring: exact typed span F1 (entities), token F1 (events), official CoNLL v8.01 (coreference), exact quote span F1, joint and conditional quote/speaker B³, and exact span+label F1 (supersenses).

## Reproduction

From the private local experiment folder, after Ollama is started with its model store redirected locally:

```bash
cd /home/drew/booknlp_llm_experiment
export HOME="$PWD/runtime/ollama/home"
export OLLAMA_MODELS="$PWD/models/ollama"
export PYTHONPATH="$PWD/src"
runtime/ollama/bin/ollama serve
```

In another terminal, pull the added model inside that store and launch a validation sample:

```bash
cd /home/drew/booknlp_llm_experiment
export HOME="$PWD/runtime/ollama/home" OLLAMA_MODELS="$PWD/models/ollama" PYTHONPATH="$PWD/src"
runtime/ollama/bin/ollama pull qwen3.5:0.8b
python external/Auto-Scalable-Speaker-Attribution-dataset/benchmark-results/booknlp-local-llm-exhaustive-2026/scripts/exhaustive.py \
  --model qwen3.5:0.8b --task events --split validation --limit 1
```

For the full frozen test cohort, change `--split validation` to `--split test` and omit `--limit`. Run one task/model at a time on this 12 GB GPU. Supported tasks: `entities`, `events`, `coref`, `quotes`, `speakers`, `supersenses`. Re-running resumes completed documents and reuses atomic response caches. Test predictions and private passage-dependent caches live under `results/exhaustive/` in the experiment folder; only aggregate metrics/configuration are suitable for publication.

To run the full test matrix sequentially, allowing each task/model result to finish before the next and refreshing aggregate reports after each row:

```bash
cd /home/drew/booknlp_llm_experiment
bash external/Auto-Scalable-Speaker-Attribution-dataset/benchmark-results/booknlp-local-llm-exhaustive-2026/scripts/run_full_matrix.sh
```

After a run, regenerate the public aggregate comparison and local category-level failure summary with:

```bash
cd /home/drew/booknlp_llm_experiment
.venv/bin/python external/Auto-Scalable-Speaker-Attribution-dataset/benchmark-results/booknlp-local-llm-exhaustive-2026/scripts/aggregate.py
.venv/bin/python external/Auto-Scalable-Speaker-Attribution-dataset/benchmark-results/booknlp-local-llm-exhaustive-2026/scripts/error_analysis.py
```

Original project tests: `cd /home/drew/booknlp_llm_experiment && .venv/bin/python -m pytest -q`. Exhaustive runner tests: see `tests/` in this directory (run with the experiment `src` on `PYTHONPATH`).

## Results

See [`RESULTS.md`](RESULTS.md). A row is included only when measured on a documented number of held-out documents. Validation pilots are labeled as validation and are not compared as final scores. Invalid batches remain represented as negative/none decisions and are reported. The model work is underway; this README intentionally makes no claim yet about whether any model improves or is a suitable synthetic-label teacher.

## Licensing and privacy

No book text, gold labels, model weights, or per-document generations are checked in. This folder contains code, prompts, model metadata, split references, aggregate scores, and documentation. Respect LitBank, SemCor/WordNet, and model licenses when reproducing or publishing results.
