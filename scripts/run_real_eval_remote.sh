#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/carr-opensource

mkdir -p outputs/real_datasets_results/darl_learnable
mkdir -p outputs/real_datasets_results/llm_baselines

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
FORCE_RERUN="${FORCE_RERUN:-0}"

# Phase 1: DARL learnable on each real dataset
for d in ML-1M Beauty Toys Steam; do
  data="data/datasets/${d}/interactions.tsv"
  out="outputs/real_datasets_results/darl_learnable/${d}"
  best_ckpt="${out}/checkpoints/best_model.pt"

  if [[ ! -f "${data}" ]]; then
    echo "[DARL][SKIP] missing data file: ${data}"
    continue
  fi

  if [[ "${FORCE_RERUN}" != "1" && -f "${best_ckpt}" ]]; then
    echo "[DARL][SKIP] ${d} already has best checkpoint: ${best_ckpt}"
    continue
  fi

  resume_args=()
  latest_ckpt="$(ls -1 "${out}"/checkpoints/ckpt_epoch*.pt 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "${latest_ckpt}" ]]; then
    resume_args+=("--resume=${latest_ckpt}")
    echo "[DARL] Resuming ${d} from ${latest_ckpt}"
  fi

  echo "[DARL] dataset=${d} data=${data} out=${out}"
  .venv/bin/python examples/mini_trainer.py \
    --data_path="${data}" \
    --out_dir="${out}" \
    --n_epochs="${EPOCHS}" \
    --batch_size="${BATCH_SIZE}" \
    "${resume_args[@]}"
done

  # Phase 1b: Post-run summary figure for learnable-vs-fixed deltas with CI
  echo "[FIGURE] Generating learnable-vs-fixed HR@10 delta figure with CI"
  .venv/bin/python scripts/generate_learnable_delta_figure.py \
    --fixed_metrics_csv outputs/real_datasets_results/darl_fixed_eval_metrics.csv

# Phase 2: LLM baselines (current script evaluates synthetic loaders)
for d in ML-1M Beauty Toys Steam; do
  out="outputs/real_datasets_results/llm_baselines/${d}"
  out_csv="${out}/llm_baselines_results.csv"
  if [[ "${FORCE_RERUN}" != "1" && -f "${out_csv}" ]]; then
    echo "[LLM][SKIP] ${d} already has results: ${out_csv}"
    continue
  fi
  echo "[LLM] dataset_tag=${d} out=${out}"
  .venv/bin/python examples/llm_baseline_evaluation.py \
    --out_dir="${out}" \
    --max_batches=20 \
    --models gpt-4o gpt-4o-mini mistral-7b qwen-7b
done

echo "[DONE] real evaluation pipeline completed"
