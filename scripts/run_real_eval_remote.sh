#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/carr-opensource

mkdir -p outputs/real_datasets_results/darl_learnable
mkdir -p outputs/real_datasets_results/llm_baselines

# Phase 1: DARL learnable on each real dataset
for d in ML-1M Beauty Toys Steam; do
  data="data/datasets/${d}/interactions.tsv"
  out="outputs/real_datasets_results/darl_learnable/${d}"
  echo "[DARL] dataset=${d} data=${data} out=${out}"
  .venv/bin/python examples/mini_trainer.py \
    --data_path="${data}" \
    --out_dir="${out}" \
    --n_epochs=30 \
    --batch_size=32
done

  # Phase 1b: Post-run summary figure for learnable-vs-fixed deltas with CI
  echo "[FIGURE] Generating learnable-vs-fixed HR@10 delta figure with CI"
  .venv/bin/python scripts/generate_learnable_delta_figure.py

# Phase 2: LLM baselines (current script evaluates synthetic loaders)
for d in ML-1M Beauty Toys Steam; do
  out="outputs/real_datasets_results/llm_baselines/${d}"
  echo "[LLM] dataset_tag=${d} out=${out}"
  .venv/bin/python examples/llm_baseline_evaluation.py \
    --out_dir="${out}" \
    --max_batches=20 \
    --models gpt-4o gpt-4o-mini mistral-7b qwen-7b
done

echo "[DONE] real evaluation pipeline completed"
