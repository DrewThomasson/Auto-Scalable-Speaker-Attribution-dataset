#!/usr/bin/env bash
# Sequential/resumable full held-out matrix. State remains in the local experiment.
set -uo pipefail
cd /home/drew/booknlp_llm_experiment
export HOME="$PWD/runtime/ollama/home"
export OLLAMA_MODELS="$PWD/models/ollama"
export OLLAMA_NUM_PARALLEL=1
export PYTHONPATH="$PWD/src"
RUNNER=external/Auto-Scalable-Speaker-Attribution-dataset/benchmark-results/booknlp-local-llm-exhaustive-2026/scripts/exhaustive.py
AGG=external/Auto-Scalable-Speaker-Attribution-dataset/benchmark-results/booknlp-local-llm-exhaustive-2026/scripts/aggregate.py
MODELS=(qwen3.5:0.8b qwen3.5:4b qwen3.5:9b gemma4:12b)
TASKS=(events entities quotes speakers supersenses coref)
mkdir -p logs results/exhaustive/logs
for task in "${TASKS[@]}"; do
  for model in "${MODELS[@]}"; do
    stamp=$(date -Is)
    logfile="logs/exhaustive_${model//:/_}_${task}_test.log"
    echo "START $stamp $model $task" | tee -a "$logfile"
    if .venv/bin/python "$RUNNER" --model "$model" --task "$task" --split test 2>&1 | tee -a "$logfile"; then
      echo "DONE $(date -Is) $model $task" | tee -a "$logfile"
    else
      rc=$?
      echo "FAILED rc=$rc $(date -Is) $model $task; rerun this script to resume" | tee -a "$logfile"
    fi
    .venv/bin/python "$AGG" || true
  done
done
