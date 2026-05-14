#!/usr/bin/env python3
"""Generate learnable-vs-fixed per-dataset HR@10 delta with confidence intervals.

This script is intended to run after the learnable sweep finishes.

Inputs expected:
  - Learnable checkpoints at:
      outputs/real_datasets_results/darl_learnable/<dataset>/checkpoints/best_model.pt
  - Real dataset files at:
      data/datasets/<dataset>/interactions.tsv

Outputs:
  - outputs/real_datasets_results/darl_learnable_eval_metrics.json
  - outputs/real_datasets_results/darl_learnable_eval_metrics.csv
  - outputs/figures/learnable_vs_fixed_hr_delta_ci.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from carr_v2.backbone import CARRBackbone
from carr_v2.data import make_loaders


# Fixed DARL values currently reported in Table main.
FIXED_HR10 = {
    "ML-1M": 0.0652,
    "Beauty": 0.0098,
    "Toys": 0.0158,
    "Steam": 0.0105,
}


def load_fixed_hr10_from_csv(path: Path) -> dict[str, float]:
    """Load per-dataset fixed HR@10 from CSV.

    Expected columns:
      - dataset (required)
      - hr_at_10 or fixed_hr10 (required)
    """
    out: dict[str, float] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = str(row.get("dataset", "")).strip()
            if not dataset:
                continue
            value = row.get("fixed_hr10", row.get("hr_at_10", None))
            if value is None or str(value).strip() == "":
                continue
            out[dataset] = float(value)
    return out


def wilson_ci(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = hits / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    radius = (z / denom) * np.sqrt((p * (1.0 - p) / total) + (z * z) / (4.0 * total * total))
    return max(0.0, center - radius), min(1.0, center + radius)


def evaluate_checkpoint(
    dataset_name: str,
    fixed_hr10: float,
    data_path: Path,
    checkpoint_path: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    top_k: int,
) -> dict[str, float | int | str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, n_items = make_loaders(
        data_path=str(data_path),
        n_users=0,
        n_items=0,
        seq_len=50,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    model = CARRBackbone(
        num_items=n_items,
        d_model=256,
        n_heads=8,
        n_layers=12,
        max_seq=60,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    hits = 0
    total = 0
    ndcg_sum = 0.0

    with torch.no_grad():
        for seqs, targets in val_loader:
            seqs = seqs.to(device)
            targets = targets.to(device)
            logits, _ = model(seqs)
            topk = logits.topk(top_k, dim=-1).indices

            match = topk == targets.unsqueeze(1)
            batch_hits = match.any(dim=1)
            hits += int(batch_hits.sum().item())
            total += int(targets.shape[0])

            for row in range(topk.size(0)):
                hit_pos = torch.where(match[row])[0]
                if hit_pos.numel() > 0:
                    rank = int(hit_pos[0].item()) + 1
                    ndcg_sum += 1.0 / np.log2(rank + 1.0)

    hr = (hits / total) if total > 0 else 0.0
    ndcg = (ndcg_sum / total) if total > 0 else 0.0
    ci_low, ci_high = wilson_ci(hits, total)

    fixed = fixed_hr10
    return {
        "dataset": dataset_name,
        "learnable_hr10": hr,
        "learnable_ndcg10": ndcg,
        "learnable_hits": hits,
        "num_samples": total,
        "learnable_ci_low": ci_low,
        "learnable_ci_high": ci_high,
        "fixed_hr10": fixed,
        "delta_hr10": hr - fixed,
        "delta_ci_low": ci_low - fixed,
        "delta_ci_high": ci_high - fixed,
        "checkpoint": str(checkpoint_path),
        "data_path": str(data_path),
    }


def plot_delta(metrics: list[dict[str, float | int | str]], out_path: Path) -> None:
    order = ["ML-1M", "Beauty", "Toys", "Steam"]
    rows = [m for d in order for m in metrics if m["dataset"] == d]
    if not rows:
        raise ValueError("No metrics available to plot.")

    datasets = [str(r["dataset"]) for r in rows]
    deltas = np.array([float(r["delta_hr10"]) for r in rows], dtype=float)
    low = np.array([float(r["delta_ci_low"]) for r in rows], dtype=float)
    high = np.array([float(r["delta_ci_high"]) for r in rows], dtype=float)

    err_low = deltas - low
    err_high = high - deltas

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    x = np.arange(len(datasets), dtype=float)
    bars = ax.bar(
        x,
        deltas,
        color="#1f77b4",
        alpha=0.88,
        edgecolor="black",
        linewidth=0.7,
    )
    ax.errorbar(
        x,
        deltas,
        yerr=np.vstack([err_low, err_high]),
        fmt="none",
        ecolor="#111111",
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
    )

    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Delta HR@10 (Learnable - Fixed)")
    ax.set_title("Per-Dataset Learnable Gain with 95% CI")
    ax.grid(axis="y", alpha=0.25)

    for bar, row in zip(bars, rows):
        n = int(row["num_samples"])
        y = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y + (0.004 if y >= 0 else -0.012),
            f"n={n}",
            ha="center",
            va="bottom" if y >= 0 else "top",
            fontsize=9,
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate learnable-vs-fixed delta figure")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ML-1M", "Beauty", "Toys", "Steam"],
        help="Datasets to evaluate",
    )
    parser.add_argument("--data_root", type=Path, default=Path("data/datasets"))
    parser.add_argument(
        "--learnable_root",
        type=Path,
        default=Path("outputs/real_datasets_results/darl_learnable"),
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--metrics_json",
        type=Path,
        default=Path("outputs/real_datasets_results/darl_learnable_eval_metrics.json"),
    )
    parser.add_argument(
        "--metrics_csv",
        type=Path,
        default=Path("outputs/real_datasets_results/darl_learnable_eval_metrics.csv"),
    )
    parser.add_argument(
        "--figure_out",
        type=Path,
        default=Path("outputs/figures/learnable_vs_fixed_hr_delta_ci.png"),
    )
    parser.add_argument(
        "--fixed_metrics_csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV with protocol-matched fixed DARL HR@10 values. "
            "Columns: dataset and (hr_at_10 or fixed_hr10)."
        ),
    )
    parser.add_argument(
        "--warn_delta_abs",
        type=float,
        default=0.25,
        help="Warn if |learnable-fixed| exceeds this threshold.",
    )
    args = parser.parse_args()

    fixed_hr10_map = dict(FIXED_HR10)
    fixed_source = "table_constants"
    if args.fixed_metrics_csv is not None:
        if not args.fixed_metrics_csv.exists():
            raise FileNotFoundError(f"--fixed_metrics_csv not found: {args.fixed_metrics_csv}")
        overrides = load_fixed_hr10_from_csv(args.fixed_metrics_csv)
        fixed_hr10_map.update(overrides)
        fixed_source = f"csv:{args.fixed_metrics_csv}"

    metrics: list[dict[str, float | int | str]] = []
    missing: list[str] = []

    for d in args.datasets:
        if d not in fixed_hr10_map:
            continue
        data_path = args.data_root / d / "interactions.tsv"
        checkpoint = args.learnable_root / d / "checkpoints" / "best_model.pt"
        if not data_path.exists() or not checkpoint.exists():
            missing.append(d)
            continue
        print(f"Evaluating learnable checkpoint for {d}...")
        metrics.append(
            evaluate_checkpoint(
                dataset_name=d,
                fixed_hr10=fixed_hr10_map[d],
                data_path=data_path,
                checkpoint_path=checkpoint,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
                top_k=args.top_k,
            )
        )

    if not metrics:
        raise FileNotFoundError(
            "No datasets had both interactions.tsv and best_model.pt. "
            "Run this after learnable training checkpoints are available."
        )

    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    for row in metrics:
        d = str(row["dataset"])
        delta = float(row["delta_hr10"])
        if abs(delta) > args.warn_delta_abs:
            warnings.append(
                f"Large |delta| for {d}: {delta:.6f}. "
                "Check protocol consistency between learnable and fixed baselines."
            )

    with open(args.metrics_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "missing": missing,
                "fixed_source": fixed_source,
                "warnings": warnings,
            },
            f,
            indent=2,
        )

    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "learnable_hr10",
                "learnable_ndcg10",
                "learnable_hits",
                "num_samples",
                "learnable_ci_low",
                "learnable_ci_high",
                "fixed_hr10",
                "delta_hr10",
                "delta_ci_low",
                "delta_ci_high",
                "checkpoint",
                "data_path",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics)

    plot_delta(metrics, args.figure_out)

    print("Saved:")
    print(args.metrics_json)
    print(args.metrics_csv)
    print(args.figure_out)
    print(f"Fixed baseline source: {fixed_source}")
    if missing:
        print("Missing datasets:", ", ".join(missing))
    if warnings:
        print("Warnings:")
        for w in warnings:
            print("-", w)


if __name__ == "__main__":
    main()
