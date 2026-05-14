#!/usr/bin/env python3
"""Generate prominent paper figures from real-data API baseline outputs.

Reads per-dataset CSV files from:
  outputs/llm_baselines_api_realdata_gpt_v3/*_results.csv

Writes:
  outputs/figures/api_realdata_hr_ndcg_grouped.png
  outputs/figures/api_realdata_relative_gain.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_ORDER = ["ML-1M", "Beauty", "Toys", "Steam"]
MODEL_ORDER = ["gpt-4o", "gpt-4o-mini"]


def load_api_results(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for dataset in DATASET_ORDER:
        csv_path = results_dir / f"{dataset}_results.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            if str(row.get("status", "")) != "success":
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "model": str(row["model"]),
                    "hr_at_10": float(row["hr_at_10"]),
                    "ndcg_at_10": float(row["ndcg_at_10"]),
                    "num_samples": int(float(row.get("num_samples", 0))),
                }
            )

    if not rows:
        raise FileNotFoundError(
            f"No successful result rows found under {results_dir}"
        )

    out = pd.DataFrame(rows)
    out["dataset"] = pd.Categorical(out["dataset"], categories=DATASET_ORDER, ordered=True)
    out["model"] = pd.Categorical(out["model"], categories=MODEL_ORDER, ordered=True)
    out = out.sort_values(["dataset", "model"]).reset_index(drop=True)
    return out


def plot_hr_ndcg_grouped(df: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
    x = np.arange(len(DATASET_ORDER))
    width = 0.35
    colors = {"gpt-4o": "#1f77b4", "gpt-4o-mini": "#ff7f0e"}

    for ax, metric, title in [
        (axes[0], "hr_at_10", "HR@10 by Dataset"),
        (axes[1], "ndcg_at_10", "NDCG@10 by Dataset"),
    ]:
        for i, model in enumerate(MODEL_ORDER):
            vals = []
            for d in DATASET_ORDER:
                sub = df[(df["dataset"] == d) & (df["model"] == model)]
                vals.append(float(sub.iloc[0][metric]) if not sub.empty else np.nan)
            ax.bar(
                x + (i - 0.5) * width,
                vals,
                width=width,
                label=model,
                color=colors[model],
                alpha=0.9,
                edgecolor="black",
                linewidth=0.6,
            )

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(DATASET_ORDER)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Score")
    axes[1].legend(loc="upper right", frameon=False)
    fig.suptitle("API LLM Real-Data Performance (Sampled-Negative Evaluation)", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_relative_gain(df: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    gains = []
    for d in DATASET_ORDER:
        hr_big = df[(df["dataset"] == d) & (df["model"] == "gpt-4o")]["hr_at_10"]
        hr_small = df[(df["dataset"] == d) & (df["model"] == "gpt-4o-mini")]["hr_at_10"]
        if hr_big.empty or hr_small.empty:
            continue
        gains.append({
            "dataset": d,
            "gain_x": float(hr_big.iloc[0] / max(hr_small.iloc[0], 1e-12)),
        })

    gdf = pd.DataFrame(gains)
    if gdf.empty:
        raise ValueError("Cannot compute relative gains; missing paired model rows.")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(
        gdf["dataset"],
        gdf["gain_x"],
        color="#2ca02c",
        alpha=0.9,
        edgecolor="black",
        linewidth=0.6,
    )
    for b, v in zip(bars, gdf["gain_x"]):
        ax.text(b.get_x() + b.get_width() / 2.0, v + 0.03, f"{v:.2f}x", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("HR@10 Ratio: GPT-4o / GPT-4o-mini")
    ax.set_title("Relative API Model Gain Across Datasets")
    ax.axhline(1.0, color="#777777", linestyle="--", linewidth=1.0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prominent real-data figures")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("outputs/llm_baselines_api_realdata_gpt_v3"),
        help="Directory with *_results.csv files",
    )
    parser.add_argument(
        "--fig_dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Directory to write figure PNG files",
    )
    args = parser.parse_args()

    df = load_api_results(args.results_dir)
    plot_hr_ndcg_grouped(df, args.fig_dir / "api_realdata_hr_ndcg_grouped.png")
    plot_relative_gain(df, args.fig_dir / "api_realdata_relative_gain.png")

    print("Saved figures:")
    print(args.fig_dir / "api_realdata_hr_ndcg_grouped.png")
    print(args.fig_dir / "api_realdata_relative_gain.png")


if __name__ == "__main__":
    main()
