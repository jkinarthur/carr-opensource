"""
CARR-v2 Ablation Runner
=======================
Compares four training policies under identical seeds at publication scale.

Policies:
  A – Fixed λ  + Fixed-ε threshold   (replicates original CARR monitoring)
  B – Fixed λ  + Learned κ           (Gumbel depth selection only)
  C – Adaptive λ + Fixed-ε threshold (simplex weighting only)
  D – Adaptive λ + Learned κ         (full CARR-v2)

Outputs:
  outputs/ablation_results.csv
  outputs/figures/ablation_bar.png

Usage on EC2:
  python examples/ablation_runner.py
  python examples/ablation_runner.py --n_users 50000 --n_items 20000 --n_epochs 100
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from carr_v2.backbone import CARRBackbone, _compute_collapse_score
from carr_v2.losses import UnifiedCompressionLoss, UnifiedLossConfig
from carr_v2.relaxations import soft_postcompression_weights, soft_precompression_reference
from carr_v2.selectors import GumbelLayerSelector
from carr_v2.weighting import AdaptiveLossWeights
from carr_v2.data import make_loaders


def _eval_hit_rate(
    model: CARRBackbone,
    loader,
    device: torch.device,
    top_k: int = 10,
) -> float:
    model.eval()
    hits = 0
    total = 0
    with torch.no_grad():
        for seqs, targets in loader:
            seqs, targets = seqs.to(device), targets.to(device)
            logits, _ = model(seqs)
            top = logits.topk(top_k, dim=-1).indices
            hits += (top == targets.unsqueeze(1)).any(dim=1).sum().item()
            total += targets.shape[0]
    return hits / max(total, 1)


def _get_layer_stats(
    model: CARRBackbone, seqs: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        _, hiddens = model(seqs.to(device))
    r_vals = torch.tensor(
        [_compute_collapse_score(h)[0] for h in hiddens], dtype=torch.float32
    )
    s_vals = torch.tensor(
        [_compute_collapse_score(h)[1] for h in hiddens], dtype=torch.float32
    )
    return r_vals, s_vals


def _fixed_threshold_kappa(
    r_layers: torch.Tensor,
    s_layers: torch.Tensor,
    eps_r: float = 0.10,
    eps_s: float = 0.05,
) -> torch.Tensor:
    for i in range(r_layers.numel()):
        if r_layers[i].item() < eps_r or s_layers[i].item() < eps_s:
            return torch.tensor(float(i))
    return torch.tensor(float(r_layers.numel() - 1))


def run_policy(
    name: str,
    use_adaptive_lambda: bool,
    use_learned_kappa: bool,
    n_epochs: int = 100,
    n_users: int = 50_000,
    n_items: int = 20_000,
    n_layers: int = 12,
    d_model: int = 256,
    n_heads: int = 8,
    seq_len: int = 50,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    num_workers: int = 4,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    train_loader, val_loader, _ = make_loaders(
        n_users=n_users, n_items=n_items, seq_len=seq_len,
        batch_size=batch_size, num_workers=num_workers, seed=seed,
    )
    model = CARRBackbone(
        num_items=n_items, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, max_seq=seq_len + 10,
    ).to(device)
    loss_fn = UnifiedCompressionLoss(UnifiedLossConfig())
    layer_weights = torch.linspace(1.0, 2.0, n_layers, device=device)
    m_star = torch.tensor(float(n_layers) * 0.667, device=device)

    selector = GumbelLayerSelector(num_layers=n_layers, init_temperature=2.0).to(device) \
        if use_learned_kappa else None
    adaptive_weights = AdaptiveLossWeights().to(device) if use_adaptive_lambda else None

    params = list(model.parameters())
    if selector:
        params += list(selector.parameters())
    if adaptive_weights:
        params += list(adaptive_weights.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    # torch.amp.GradScaler is unavailable in torch 2.2; fall back to cuda.amp.
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_losses: list[float] = []
    last_out: dict = {}

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            current_lmb = adaptive_weights() if adaptive_weights else None

        for seqs, targets in train_loader:
            seqs = seqs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits, hiddens = model(seqs)
                rec_loss = F.cross_entropy(logits, targets)

                r_layers = torch.tensor(
                    [_compute_collapse_score(h)[0] for h in hiddens],
                    dtype=torch.float32, device=device,
                )
                s_layers = torch.tensor(
                    [_compute_collapse_score(h)[1] for h in hiddens],
                    dtype=torch.float32, device=device,
                )

                if use_learned_kappa:
                    selection = selector(training=True)
                    kappa = selection.soft_index
                    entropy = selection.entropy
                else:
                    kappa = _fixed_threshold_kappa(r_layers.cpu(), s_layers.cpu()).to(device)
                    entropy = torch.tensor(0.0, device=device)

                r_pre = soft_precompression_reference(r_layers, kappa=kappa)
                post_mask = soft_postcompression_weights(num_layers=n_layers, kappa=kappa)

                last_out = loss_fn(
                    rec_loss=rec_loss,
                    r_layers=r_layers, r_pre_compress=r_pre,
                    post_mask=post_mask, layer_weights=layer_weights,
                    evidence_layers=s_layers, kappa=kappa,
                    m_star=m_star, selector_entropy=entropy,
                    num_layers=n_layers, seq_len=seq_len,
                    lambda_overrides=current_lmb,
                )

            scaler.scale(last_out["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(last_out["total"].detach())
            n_batches += 1

        train_losses.append(epoch_loss / max(n_batches, 1))

        if selector:
            selector.set_temperature(max(0.2, 2.0 * (0.97 ** epoch)))
        if adaptive_weights:
            adaptive_weights.set_temperature(max(0.1, 1.0 * (0.995 ** epoch)))

        if epoch % 20 == 0:
            print(f"  [{name}] epoch {epoch}/{n_epochs}  loss={train_losses[-1]:.4f}")

    hr = _eval_hit_rate(model, val_loader, device, top_k=10)

    val_seqs, _ = next(iter(val_loader))
    r_final, s_final = _get_layer_stats(model, val_seqs, device)

    if selector:
        with torch.no_grad():
            sel = selector(training=False)
        kappa_final = float(sel.soft_index)
    else:
        kappa_final = float(_fixed_threshold_kappa(r_final, s_final))

    if adaptive_weights:
        with torch.no_grad():
            final_lmb_vec = adaptive_weights()["vector"].tolist()
    else:
        final_lmb_vec = [0.20, 0.20, 0.05, 0.01]

    print(f"[{name}] HR@10={hr:.6f}  κ={kappa_final:.2f}  loss={train_losses[-1]:.4f}")
    return {
        "policy": name,
        "hr_at_10": hr,
        "kappa_final": kappa_final,
        "final_train_loss": train_losses[-1],
        "r_final_mean": float(r_final.mean()),
        "s_final_mean": float(s_final.mean()),
        "lam_geometry": final_lmb_vec[0],
        "lam_evidence": final_lmb_vec[1],
        "lam_efficiency": final_lmb_vec[2],
        "lam_entropy": final_lmb_vec[3],
    }


def run_all(
    out_dir: str = "outputs",
    n_epochs: int = 100,
    n_users: int = 50_000,
    n_items: int = 20_000,
    n_layers: int = 12,
    d_model: int = 256,
    n_heads: int = 8,
    seq_len: int = 50,
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 42,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    policies = [
        ("A: Fixed-λ  + Fixed-ε",    False, False),
        ("B: Fixed-λ  + Learned-κ",   False, True),
        ("C: Adaptive-λ + Fixed-ε",   True,  False),
        ("D: Adaptive-λ + Learned-κ", True,  True),
    ]

    shared = dict(
        n_epochs=n_epochs, n_users=n_users, n_items=n_items,
        n_layers=n_layers, d_model=d_model, n_heads=n_heads,
        seq_len=seq_len, batch_size=batch_size,
        num_workers=num_workers, seed=seed,
    )

    results = []
    for name, use_adap_lam, use_learned_kap in policies:
        print(f"\n=== Policy: {name} ===")
        r = run_policy(
            name=name,
            use_adaptive_lambda=use_adap_lam,
            use_learned_kappa=use_learned_kap,
            **shared,
        )
        results.append(r)

    csv_path = os.path.join(out_dir, "ablation_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nAblation table written to {csv_path}")

    _plot_ablation(results, out_dir)


def _plot_ablation(results: list[dict], out_dir: str) -> None:
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    names = [r["policy"] for r in results]
    hr_vals    = [r["hr_at_10"]      for r in results]
    r_means    = [r["r_final_mean"]  for r in results]
    s_means    = [r["s_final_mean"]  for r in results]
    losses     = [r["final_train_loss"] for r in results]

    colors = ["#cccccc", "#6baed6", "#74c476", "#2171b5"]
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    metric_sets = [
        (hr_vals,  "HR@10 (val)",            "HR@10"),
        (r_means,  "Final Mean R(l)",         "R(l) mean"),
        (s_means,  "Final Mean S_l",          "S_l mean"),
        (losses,   "Final Train Loss",        "Loss"),
    ]
    short_names = [n.split(": ")[1] if ": " in n else n for n in names]

    for ax, (vals, title, ylabel) in zip(axes, metric_sets):
        bars = ax.bar(range(len(names)), vals, color=colors, width=0.55, edgecolor="white")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(short_names, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(abs(v) * 0.01, 1e-6),
                f"{v:.4f}", ha="center", va="bottom", fontsize=8,
            )

    fig.suptitle("CARR-v2 Ablation: Policy A→D Comparison", fontsize=12, y=1.02)
    fig.tight_layout()
    out_path = os.path.join(fig_dir, "ablation_bar.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CARR-v2 Ablation Runner")
    p.add_argument("--n_epochs",    type=int, default=100)
    p.add_argument("--n_users",     type=int, default=50_000)
    p.add_argument("--n_items",     type=int, default=20_000)
    p.add_argument("--n_layers",    type=int, default=12)
    p.add_argument("--d_model",     type=int, default=256)
    p.add_argument("--n_heads",     type=int, default=8)
    p.add_argument("--seq_len",     type=int, default=50)
    p.add_argument("--batch_size",  type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_dir",     type=str, default="outputs")
    p.add_argument("--seed",        type=int, default=42)
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_all(**vars(args))

