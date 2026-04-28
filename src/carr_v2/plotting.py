"""
Publication-quality plots generated from CARR-v2 experiment CSVs.

Outputs:
  outputs/figures/lambda_trajectories.png
  outputs/figures/depth_convergence.png
  outputs/figures/kappa_vs_critical.png
"""

import csv
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def _load_csv(path: str) -> dict[str, list[float]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        result: dict[str, list] = {k: [] for k in reader.fieldnames or []}
        for row in reader:
            for k, v in row.items():
                try:
                    result[k].append(float(v))
                except (ValueError, TypeError):
                    result[k].append(v)
    return result


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    })


def plot_lambda_trajectories(csv_path: str, out_path: str) -> None:
    data = _load_csv(csv_path)
    epochs = data["epoch"]

    _style()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    labels = ["λ-geometry", "λ-evidence", "λ-efficiency", "λ-entropy"]
    keys = ["lam_geometry", "lam_evidence", "lam_efficiency", "lam_entropy"]

    for col, lab, key in zip(colors, labels, keys):
        if key in data:
            ax.plot(epochs, data[key], color=col, label=lab, linewidth=1.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Simplex weight")
    ax.set_title("Adaptive λ trajectories (bilevel meta-learning)")
    ax.legend(loc="upper right", ncol=2, frameon=False)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_depth_convergence(csv_path: str, out_path: str) -> None:
    data = _load_csv(csv_path)
    epochs = data["epoch"]

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: soft depth and critical depth over epochs.
    ax = axes[0]
    ax.plot(epochs, data["soft_depth"], color="#1f77b4", label="κ (learned soft depth)", linewidth=1.8)
    ax.axhline(y=data["critical_depth"][0], color="#d62728", linestyle="--", linewidth=1.5, label=f"k* proxy = {data['critical_depth'][0]:.1f}")
    ax.fill_between(epochs, data["soft_depth"], data["critical_depth"][0], alpha=0.12, color="#1f77b4")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Compression depth")
    ax.set_title("Learned κ vs Critical Depth k*")
    ax.legend(frameon=False)

    # Right: absolute gap over epochs.
    ax2 = axes[1]
    ax2.plot(epochs, data["abs_gap"], color="#ff7f0e", linewidth=1.8)
    ax2.axhline(y=0.5, color="#aaaaaa", linestyle=":", linewidth=1.2, label="tol = 0.5")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("|κ − k*|")
    ax2.set_title("Convergence Gap Over Training")
    ax2.legend(frameon=False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_train_val_loss(csv_path: str, out_path: str) -> None:
    data = _load_csv(csv_path)
    if "train_loss" not in data:
        return
    epochs = data["epoch"]

    _style()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(epochs, data["train_loss"], color="#1f77b4", label="Train loss", linewidth=1.8)
    ax.plot(epochs, data["val_loss"], color="#ff7f0e", linestyle="--", label="Val loss", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Unified loss")
    ax.set_title("Train / Validation Loss (CARR-v2)")
    ax.legend(frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_all(out_dir: str = "outputs") -> None:
    fig_dir = os.path.join(out_dir, "figures")

    lambda_csv = os.path.join(out_dir, "mini_trainer_lambdas.csv")
    depth_csv = os.path.join(out_dir, "mini_trainer_depth_convergence.csv")
    bilevel_depth_csv = os.path.join(out_dir, "depth_convergence.csv")

    if os.path.exists(lambda_csv):
        plot_lambda_trajectories(lambda_csv, os.path.join(fig_dir, "lambda_trajectories.png"))
        plot_train_val_loss(lambda_csv, os.path.join(fig_dir, "train_val_loss.png"))

    if os.path.exists(depth_csv):
        plot_depth_convergence(depth_csv, os.path.join(fig_dir, "depth_convergence.png"))
    elif os.path.exists(bilevel_depth_csv):
        plot_depth_convergence(bilevel_depth_csv, os.path.join(fig_dir, "depth_convergence.png"))


if __name__ == "__main__":
    generate_all()
