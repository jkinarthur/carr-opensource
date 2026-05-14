#!/usr/bin/env python3
"""Train/evaluate fixed-DARL baselines on real datasets and export HR@10 metrics.

This script creates protocol-matched fixed baselines for
scripts/generate_learnable_delta_figure.py via --fixed_metrics_csv.

Outputs:
  - outputs/real_datasets_results/darl_fixed_eval_metrics.csv
  - outputs/real_datasets_results/darl_fixed_eval_metrics.json
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


def run_fixed_for_dataset(
    dataset: str,
    data_path: Path,
    out_root: Path,
    n_epochs: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    top_k: int,
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

    model = CARRBackbone(
        num_items=n_items,
        d_model=256,
        n_heads=8,
        n_layers=12,
        max_seq=60,
    ).to(device)
    loss_fn = UnifiedCompressionLoss(UnifiedLossConfig())
    layer_weights = torch.linspace(1.0, 2.0, 12, device=device)
    m_star = torch.tensor(float(12) * 0.667, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val = -1.0
    best_path = out_root / dataset / "checkpoints" / "best_model.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

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

                kappa = fixed_threshold_kappa(r_layers, s_layers)
                post_mask = soft_postcompression_weights(num_layers=12, kappa=kappa)
                r_pre = soft_precompression_reference(r_layers, kappa=kappa)

                out = loss_fn(
                    rec_loss=rec_loss,
                    r_layers=r_layers,
                    r_pre_compress=r_pre,
                    post_mask=post_mask,
                    layer_weights=layer_weights,
                    evidence_layers=s_layers,
                    kappa=kappa,
                    m_star=m_star,
                    selector_entropy=torch.tensor(0.0, device=device),
                    num_layers=12,
                    seq_len=50,
                    lambda_overrides=None,
                )

            scaler.scale(out["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        hr, _, _ = eval_hr_ndcg(model, val_loader, device, top_k=top_k)
        if hr > best_val:
            best_val = hr
            torch.save(model.state_dict(), best_path)

        if epoch % 10 == 0 or epoch == 1 or epoch == n_epochs:
            print(f"[{dataset}] epoch {epoch}/{n_epochs} best_hr={best_val:.6f}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    hr, ndcg, n_samples = eval_hr_ndcg(model, val_loader, device, top_k=top_k)
    return {
        "dataset": dataset,
        "hr_at_10": hr,
        "ndcg_at_10": ndcg,
        "num_samples": n_samples,
        "checkpoint": str(best_path),
        "data_path": str(data_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed DARL real-data metrics")
    parser.add_argument("--datasets", nargs="+", default=["ML-1M", "Beauty", "Toys", "Steam"])
    parser.add_argument("--data_root", type=Path, default=Path("data/datasets"))
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("outputs/real_datasets_results/darl_fixed"),
    )
    parser.add_argument(
        "--metrics_csv",
        type=Path,
        default=Path("outputs/real_datasets_results/darl_fixed_eval_metrics.csv"),
    )
    parser.add_argument(
        "--metrics_json",
        type=Path,
        default=Path("outputs/real_datasets_results/darl_fixed_eval_metrics.json"),
    )
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    rows: list[dict[str, float | int | str]] = []
    missing: list[str] = []
    for d in args.datasets:
        data_path = args.data_root / d / "interactions.tsv"
        if not data_path.exists():
            missing.append(d)
            continue
        print(f"Running fixed DARL for {d}...")
        rows.append(
            run_fixed_for_dataset(
                dataset=d,
                data_path=data_path,
                out_root=args.out_root,
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
                top_k=args.top_k,
            )
        )

    if not rows:
        raise FileNotFoundError("No datasets were processed. Check data_root and --datasets.")

    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "hr_at_10", "ndcg_at_10", "num_samples", "checkpoint", "data_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_json, "w", encoding="utf-8") as f:
        json.dump({"metrics": rows, "missing": missing}, f, indent=2)

    print("Saved:")
    print(args.metrics_csv)
    print(args.metrics_json)
    if missing:
        print("Missing datasets:", ", ".join(missing))


if __name__ == "__main__":
    main()
