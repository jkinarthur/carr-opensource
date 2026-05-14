#!/usr/bin/env python3
"""Run missing ablation policies B/C on real datasets.

Policy B: Fixed lambda + learned kappa
Policy C: Adaptive lambda + fixed-threshold kappa

Outputs:
  outputs/real_datasets_results/ablations/ablation_bc_results.csv
  outputs/real_datasets_results/ablations/ablation_bc_results.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from carr_v2.backbone import CARRBackbone, _compute_collapse_score
from carr_v2.data import make_loaders
from carr_v2.losses import UnifiedCompressionLoss, UnifiedLossConfig
from carr_v2.relaxations import soft_postcompression_weights, soft_precompression_reference
from carr_v2.selectors import GumbelLayerSelector
from carr_v2.weighting import AdaptiveLossWeights


def fixed_threshold_kappa(
    r_layers: torch.Tensor,
    s_layers: torch.Tensor,
    eps_r: float = 0.10,
    eps_s: float = 0.05,
) -> torch.Tensor:
    for i in range(r_layers.numel()):
        if r_layers[i].item() < eps_r or s_layers[i].item() < eps_s:
            return torch.tensor(float(i), device=r_layers.device)
    return torch.tensor(float(r_layers.numel() - 1), device=r_layers.device)


def eval_hr_ndcg(model: CARRBackbone, loader, device: torch.device, top_k: int = 10) -> tuple[float, float, int]:
    model.eval()
    hits = 0
    total = 0
    ndcg_sum = 0.0
    with torch.no_grad():
        for seqs, targets in loader:
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
    return hr, ndcg, total


def run_dataset_policy(
    policy: str,
    dataset: str,
    data_path: Path,
    out_root: Path,
    n_epochs: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> dict[str, float | int | str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    torch.manual_seed(seed)

    train_loader, val_loader, n_items = make_loaders(
        data_path=str(data_path),
        n_users=0,
        n_items=0,
        seq_len=50,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    model = CARRBackbone(num_items=n_items, d_model=256, n_heads=8, n_layers=12, max_seq=60).to(device)
    loss_fn = UnifiedCompressionLoss(UnifiedLossConfig())
    layer_weights = torch.linspace(1.0, 2.0, 12, device=device)
    m_star = torch.tensor(float(12) * 0.667, device=device)

    selector = GumbelLayerSelector(num_layers=12, init_temperature=2.0).to(device)
    adaptive = AdaptiveLossWeights().to(device)

    fixed_lambda = {
        "geometry": torch.tensor(0.2, device=device),
        "evidence": torch.tensor(0.2, device=device),
        "efficiency": torch.tensor(0.05, device=device),
        "entropy": torch.tensor(0.01, device=device),
    }

    # Policy setup
    if policy == "B":
        use_learned_kappa = True
        use_adaptive_lambda = False
        params = list(model.parameters()) + list(selector.parameters())
    elif policy == "C":
        use_learned_kappa = False
        use_adaptive_lambda = True
        params = list(model.parameters()) + list(adaptive.parameters())
    else:
        raise ValueError(f"Unknown policy: {policy}")

    optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-2)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_hr = -1.0
    ckpt_path = out_root / dataset / f"policy_{policy}" / "checkpoints" / "best_model.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, n_epochs + 1):
        model.train()
        for seqs, targets in train_loader:
            seqs = seqs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits, hiddens = model(seqs)
                rec_loss = F.cross_entropy(logits, targets)

                r_layers = torch.tensor(
                    [_compute_collapse_score(h)[0] for h in hiddens],
                    dtype=torch.float32,
                    device=device,
                )
                s_layers = torch.tensor(
                    [_compute_collapse_score(h)[1] for h in hiddens],
                    dtype=torch.float32,
                    device=device,
                )

                if use_learned_kappa:
                    sel = selector(training=True)
                    kappa = sel.soft_index
                    entropy = sel.entropy
                else:
                    kappa = fixed_threshold_kappa(r_layers, s_layers)
                    entropy = torch.tensor(0.0, device=device)

                lmb = adaptive() if use_adaptive_lambda else fixed_lambda

                out = loss_fn(
                    rec_loss=rec_loss,
                    r_layers=r_layers,
                    r_pre_compress=soft_precompression_reference(r_layers, kappa=kappa),
                    post_mask=soft_postcompression_weights(num_layers=12, kappa=kappa),
                    layer_weights=layer_weights,
                    evidence_layers=s_layers,
                    kappa=kappa,
                    m_star=m_star,
                    selector_entropy=entropy,
                    num_layers=12,
                    seq_len=50,
                    lambda_overrides=lmb,
                )

            scaler.scale(out["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        if use_learned_kappa:
            selector.set_temperature(max(0.2, 2.0 * (0.97 ** epoch)))
        if use_adaptive_lambda:
            adaptive.set_temperature(max(0.1, 1.0 * (0.995 ** epoch)))

        hr, _, _ = eval_hr_ndcg(model, val_loader, device, top_k=10)
        if hr > best_val_hr:
            best_val_hr = hr
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 10 == 0 or epoch == 1 or epoch == n_epochs:
            print(f"[{dataset}][Policy {policy}] epoch {epoch}/{n_epochs} best_hr={best_val_hr:.6f}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    hr, ndcg, n_samples = eval_hr_ndcg(model, val_loader, device, top_k=10)

    # Compute final proxy stats on one validation batch.
    val_seqs, _ = next(iter(val_loader))
    with torch.no_grad():
        _, hiddens = model(val_seqs.to(device))
    r_layers = torch.tensor([_compute_collapse_score(h)[0] for h in hiddens], dtype=torch.float32)
    s_layers = torch.tensor([_compute_collapse_score(h)[1] for h in hiddens], dtype=torch.float32)

    if use_learned_kappa:
        with torch.no_grad():
            sel_eval = selector(training=False)
        kappa_final = float(sel_eval.soft_index.item())
        lam_vec = [0.2, 0.2, 0.05, 0.01]
    else:
        kappa_final = float(fixed_threshold_kappa(r_layers.to(device), s_layers.to(device)).item())
        with torch.no_grad():
            vec = adaptive()["vector"].detach().cpu().numpy().tolist()
        lam_vec = [float(x) for x in vec]

    return {
        "dataset": dataset,
        "policy": policy,
        "hr_at_10": float(hr),
        "ndcg_at_10": float(ndcg),
        "num_samples": int(n_samples),
        "r_final_mean": float(r_layers.mean().item()),
        "s_final_mean": float(s_layers.mean().item()),
        "kappa_final": float(kappa_final),
        "lam_geometry": lam_vec[0],
        "lam_evidence": lam_vec[1],
        "lam_efficiency": lam_vec[2],
        "lam_entropy": lam_vec[3],
        "checkpoint": str(ckpt_path),
        "data_path": str(data_path),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Run missing real-data ablations (B/C)")
    p.add_argument("--datasets", nargs="+", default=["ML-1M", "Beauty", "Toys", "Steam"])
    p.add_argument("--policies", nargs="+", default=["B", "C"])
    p.add_argument("--data_root", type=Path, default=Path("data/datasets"))
    p.add_argument(
        "--out_root",
        type=Path,
        default=Path("outputs/real_datasets_results/ablations"),
    )
    p.add_argument("--n_epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rows: list[dict[str, float | int | str]] = []
    missing: list[str] = []

    for d in args.datasets:
        data_path = args.data_root / d / "interactions.tsv"
        if not data_path.exists():
            missing.append(d)
            continue
        for p_name in args.policies:
            print(f"Running policy {p_name} on {d}...")
            rows.append(
                run_dataset_policy(
                    policy=p_name,
                    dataset=d,
                    data_path=data_path,
                    out_root=args.out_root,
                    n_epochs=args.n_epochs,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    seed=args.seed,
                )
            )

    if not rows:
        raise RuntimeError("No ablation runs completed.")

    args.out_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_root / "ablation_bc_results.csv"
    json_path = args.out_root / "ablation_bc_results.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "missing_datasets": missing}, f, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    if missing:
        print(f"Missing datasets: {missing}")


if __name__ == "__main__":
    main()
