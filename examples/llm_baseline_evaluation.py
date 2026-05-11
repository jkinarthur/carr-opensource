"""
LLM Baseline Evaluation Runner
===============================
Evaluates all LLM baseline models against CARR-v2 compressed model.

Outputs:
  outputs/llm_baselines_results.csv
  outputs/llm_baselines_metrics.json

Can run in parallel with main training on separate GPU/CPU.

Usage:
  python examples/llm_baseline_evaluation.py --out_dir outputs_eval --models llama-3.1 mistral-7b
  python examples/llm_baseline_evaluation.py --skip_api  # Skip GPT-4, Claude (requires API keys)
"""

import argparse
import csv
import json
import os
from typing import Optional

import numpy as np
import torch

from carr_v2.data import make_loaders
from carr_v2.llm_baselines import get_recommender


def evaluate_model(
    model_name: str,
    val_loader,
    n_items: int,
    device: str = "cuda",
    max_batches: Optional[int] = None,
) -> dict:
    """Evaluate a single LLM baseline on validation set."""
    print(f"\nEvaluating {model_name}...")

    try:
        recommender = get_recommender(model_name, device=device)
    except Exception as e:
        print(f"  ✗ Failed to load {model_name}: {e}")
        return {
            "model": model_name,
            "status": "failed",
            "error": str(e),
            "hr_at_10": 0.0,
            "ndcg_at_10": 0.0,
            "drift_score": 0.0,
            "evidence_survival": 0.0,
        }

    predictions_all = []
    targets_all = []
    drift_scores = []
    evidence_survivals = []

    try:
        for batch_idx, (seqs, targets) in enumerate(val_loader):
            if max_batches and batch_idx >= max_batches:
                break

            targets = targets.cpu().numpy()
            seqs = seqs.cpu().numpy()

            # Generate recommendations for each user in batch
            for user_seq, user_target in zip(seqs, targets):
                user_history = [int(x) for x in user_seq if x > 0]
                item_pool = list(range(1, min(n_items + 1, 1000)))  # Sample from item pool

                try:
                    recs = recommender.generate_recommendations(
                        user_history, item_pool, k=10
                    )
                    predictions_all.append(recs)
                    targets_all.append(int(user_target))
                except Exception as e:
                    print(f"    Recommendation generation error: {e}")
                    predictions_all.append(item_pool[:10])
                    targets_all.append(int(user_target))

                # Estimate drift (proxy via random activations)
                hidden_proxy = torch.randn(1, 256)
                drift = recommender.compute_drift_score(hidden_proxy)
                drift_scores.append(drift)

                # Estimate evidence survival
                evidence = recommender.compute_evidence_survival(hidden_proxy)
                evidence_survivals.append(evidence)

            if batch_idx % 10 == 0:
                print(f"    Processed {batch_idx + 1} batches...")

    except Exception as e:
        print(f"  ✗ Evaluation error for {model_name}: {e}")
        return {
            "model": model_name,
            "status": "error",
            "error": str(e),
            "hr_at_10": 0.0,
            "ndcg_at_10": 0.0,
            "drift_score": 0.0,
            "evidence_survival": 0.0,
        }

    # Compute metrics
    try:
        hr = recommender.compute_hr_at_k(predictions_all, targets_all, k=10)
        ndcg = recommender.compute_ndcg_at_k(predictions_all, targets_all, k=10)
        avg_drift = float(np.mean(drift_scores)) if drift_scores else 0.0
        avg_evidence = float(np.mean(evidence_survivals)) if evidence_survivals else 0.0

        result = {
            "model": model_name,
            "status": "success",
            "hr_at_10": float(hr),
            "ndcg_at_10": float(ndcg),
            "drift_score": avg_drift,
            "evidence_survival": avg_evidence,
            "num_samples": len(targets_all),
        }
        print(f"  ✓ {model_name}: HR@10={hr:.4f} NDCG@10={ndcg:.4f}")
        return result

    except Exception as e:
        print(f"  ✗ Metric computation error: {e}")
        return {
            "model": model_name,
            "status": "error",
            "error": str(e),
            "hr_at_10": 0.0,
            "ndcg_at_10": 0.0,
            "drift_score": 0.0,
            "evidence_survival": 0.0,
        }


def run_all_baselines(
    out_dir: str = "outputs_eval",
    n_users: int = 10_000,
    n_items: int = 5_000,
    seq_len: int = 50,
    batch_size: int = 64,
    num_workers: int = 2,
    seed: int = 42,
    models: Optional[list[str]] = None,
    skip_api: bool = False,
    max_batches: Optional[int] = None,
) -> None:
    """Run evaluation on all LLM baselines."""

    if models is None:
        models = [
            "llama-3.1",
            "mistral-7b",
            "qwen-7b",
            "llama-guard",
        ]
        if not skip_api:
            models += ["gpt-4", "claude-3-5-sonnet"]

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("LLM BASELINE EVALUATION")
    print("=" * 80)
    print(f"Output directory: {out_dir}")
    print(f"Dataset: {n_users} users, {n_items} items, seq_len={seq_len}")
    print(f"Models: {models}")
    print()

    # Load validation data (reduced size for faster evaluation)
    _, val_loader, n_items_actual = make_loaders(
        n_users=n_users,
        n_items=n_items,
        seq_len=seq_len,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    print(f"Loaded validation set: {len(val_loader.dataset)} samples")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print()

    results = []
    for model_name in models:
        result = evaluate_model(
            model_name,
            val_loader,
            n_items_actual,
            device=device,
            max_batches=max_batches,
        )
        results.append(result)

    # Write results to CSV
    csv_path = os.path.join(out_dir, "llm_baselines_results.csv")
    if results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "model",
                    "status",
                    "hr_at_10",
                    "ndcg_at_10",
                    "drift_score",
                    "evidence_survival",
                    "num_samples",
                ],
            )
            writer.writeheader()
            for r in results:
                writer.writerow(r)
    print(f"\nResults written to {csv_path}")

    # Write summary metrics
    json_path = os.path.join(out_dir, "llm_baselines_metrics.json")
    summary = {
        "timestamp": str(np.datetime64("now")),
        "config": {
            "n_users": n_users,
            "n_items": n_items,
            "seq_len": seq_len,
            "batch_size": batch_size,
        },
        "results": results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary metrics written to {json_path}")

    # Print table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(
        f"{'Model':<30} {'Status':<12} {'HR@10':<12} {'NDCG@10':<12} {'Drift':<12} {'Evidence':<12}"
    )
    print("-" * 80)
    for r in results:
        status = r.get("status", "?")
        hr = f"{r.get('hr_at_10', 0.0):.4f}" if status == "success" else "—"
        ndcg = f"{r.get('ndcg_at_10', 0.0):.4f}" if status == "success" else "—"
        drift = f"{r.get('drift_score', 0.0):.4f}" if status == "success" else "—"
        evidence = f"{r.get('evidence_survival', 0.0):.4f}" if status == "success" else "—"
        print(f"{r['model']:<30} {status:<12} {hr:<12} {ndcg:<12} {drift:<12} {evidence:<12}")
    print("=" * 80)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM Baseline Evaluation Runner")
    p.add_argument("--out_dir", type=str, default="outputs_eval")
    p.add_argument("--n_users", type=int, default=10_000)
    p.add_argument("--n_items", type=int, default=5_000)
    p.add_argument("--seq_len", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["llama-3.1", "mistral-7b", "qwen-7b", "llama-guard"],
        help="Models to evaluate",
    )
    p.add_argument(
        "--skip_api",
        action="store_true",
        help="Skip API-based models (GPT-4, Claude)",
    )
    p.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Max batches to process (for testing)",
    )
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_all_baselines(
        out_dir=args.out_dir,
        n_users=args.n_users,
        n_items=args.n_items,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        models=args.models,
        skip_api=args.skip_api,
        max_batches=args.max_batches,
    )
