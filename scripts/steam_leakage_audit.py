#!/usr/bin/env python3
"""Strict leakage audit for Steam evaluation/training protocol.

Checks performed:
1) Exact train/val overlap at (sequence, target) tuple level.
2) Candidate-set integrity for sampled-negative evaluation.
3) Optional checkpoint sanity checks:
   - full-catalog HR@10 / NDCG@10
   - sampled-negative HR@10 / NDCG@10
   - shuffled-target sanity gap

Outputs:
  outputs/real_datasets_results/audits/steam_leakage_audit.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from carr_v2.backbone import CARRBackbone
from carr_v2.data import make_loaders


def _tupleize_batch(seqs: torch.Tensor, targets: torch.Tensor) -> list[tuple[tuple[int, ...], int]]:
    out: list[tuple[tuple[int, ...], int]] = []
    s_cpu = seqs.cpu().numpy()
    t_cpu = targets.cpu().numpy()
    for s, t in zip(s_cpu, t_cpu):
        out.append((tuple(int(x) for x in s.tolist()), int(t)))
    return out


def _hr_ndcg_full_catalog(model: CARRBackbone, loader, device: torch.device, top_k: int = 10) -> tuple[float, float, int]:
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
    hr = hits / total if total else 0.0
    ndcg = ndcg_sum / total if total else 0.0
    return hr, ndcg, total


def _hr_ndcg_sampled_negative(
    model: CARRBackbone,
    loader,
    n_items: int,
    device: torch.device,
    seed: int,
    neg_pool_size: int = 99,
    top_k: int = 10,
) -> tuple[float, float, int]:
    model.eval()
    rng = random.Random(seed)
    hits = 0
    total = 0
    ndcg_sum = 0.0

    all_items = list(range(n_items))

    with torch.no_grad():
        for seqs, targets in loader:
            seqs = seqs.to(device)
            targets = targets.to(device)
            logits, _ = model(seqs)

            for row in range(seqs.size(0)):
                tgt = int(targets[row].item())
                negatives = [i for i in all_items if i != tgt]
                sampled = rng.sample(negatives, k=min(neg_pool_size, len(negatives)))
                candidate = sampled + [tgt]
                rng.shuffle(candidate)

                # Strict integrity checks for this sampled set.
                if candidate.count(tgt) != 1:
                    raise RuntimeError("Target count in candidate set != 1")
                if any(x == tgt for x in sampled):
                    raise RuntimeError("Target leaked into negative pool")

                scores = logits[row, candidate]
                k = min(top_k, len(candidate))
                idx = torch.topk(scores, k=k, dim=0).indices.tolist()
                pred_items = [candidate[i] for i in idx]

                total += 1
                if tgt in pred_items:
                    hits += 1
                    rank = pred_items.index(tgt) + 1
                    ndcg_sum += 1.0 / np.log2(rank + 1.0)

    hr = hits / total if total else 0.0
    ndcg = ndcg_sum / total if total else 0.0
    return hr, ndcg, total


def _shuffled_target_sanity(model: CARRBackbone, loader, device: torch.device, seed: int, top_k: int = 10) -> tuple[float, float]:
    model.eval()
    rng = random.Random(seed)
    true_hits = 0
    shuf_hits = 0
    total = 0
    with torch.no_grad():
        for seqs, targets in loader:
            seqs = seqs.to(device)
            targets = targets.to(device)
            logits, _ = model(seqs)
            topk = logits.topk(top_k, dim=-1).indices

            batch_true = (topk == targets.unsqueeze(1)).any(dim=1)
            true_hits += int(batch_true.sum().item())

            shuffled = targets.detach().cpu().tolist()
            rng.shuffle(shuffled)
            shuffled_t = torch.tensor(shuffled, dtype=targets.dtype, device=device)
            batch_shuf = (topk == shuffled_t.unsqueeze(1)).any(dim=1)
            shuf_hits += int(batch_shuf.sum().item())
            total += int(targets.shape[0])

    true_hr = true_hits / total if total else 0.0
    shuf_hr = shuf_hits / total if total else 0.0
    return true_hr, shuf_hr


def _load_model(checkpoint: Path, n_items: int, device: torch.device) -> CARRBackbone:
    model = CARRBackbone(num_items=n_items, d_model=256, n_heads=8, n_layers=12, max_seq=60).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    return model


def main() -> None:
    p = argparse.ArgumentParser(description="Strict Steam leakage audit")
    p.add_argument("--data_path", type=Path, default=Path("data/datasets/Steam/interactions.tsv"))
    p.add_argument("--fixed_checkpoint", type=Path, default=None)
    p.add_argument("--learnable_checkpoint", type=Path, default=None)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--neg_pool_size", type=int, default=99)
    p.add_argument(
        "--out_json",
        type=Path,
        default=Path("outputs/real_datasets_results/audits/steam_leakage_audit.json"),
    )
    args = p.parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(f"Steam data not found: {args.data_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, n_items = make_loaders(
        data_path=str(args.data_path),
        n_users=0,
        n_items=0,
        seq_len=50,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # ---- Core leakage checks
    train_pairs = set()
    for seqs, targets in train_loader:
        train_pairs.update(_tupleize_batch(seqs, targets))

    val_pairs = set()
    for seqs, targets in val_loader:
        val_pairs.update(_tupleize_batch(seqs, targets))

    overlap = train_pairs.intersection(val_pairs)
    split_leakage_pass = len(overlap) == 0

    # Candidate integrity check on random batches (implemented inside sampled eval too)
    candidate_integrity_pass = True
    candidate_integrity_error = ""
    try:
        # Run with a tiny random model to execute candidate checks deterministically.
        probe = CARRBackbone(num_items=n_items, d_model=256, n_heads=8, n_layers=12, max_seq=60).to(device)
        _hr_ndcg_sampled_negative(
            probe,
            val_loader,
            n_items=n_items,
            device=device,
            seed=args.seed,
            neg_pool_size=args.neg_pool_size,
            top_k=10,
        )
    except Exception as e:  # noqa: BLE001
        candidate_integrity_pass = False
        candidate_integrity_error = str(e)

    results: dict[str, object] = {
        "data_path": str(args.data_path),
        "device": str(device),
        "n_items": int(n_items),
        "n_train_windows": int(len(train_pairs)),
        "n_val_windows": int(len(val_pairs)),
        "n_overlap_windows": int(len(overlap)),
        "split_leakage_pass": split_leakage_pass,
        "candidate_integrity_pass": candidate_integrity_pass,
        "candidate_integrity_error": candidate_integrity_error,
        "models": {},
    }

    def evaluate_checkpoint(label: str, ckpt: Path) -> None:
        model = _load_model(ckpt, n_items=n_items, device=device)
        full_hr, full_ndcg, n_full = _hr_ndcg_full_catalog(model, val_loader, device=device, top_k=10)
        sampled_hr, sampled_ndcg, n_sampled = _hr_ndcg_sampled_negative(
            model,
            val_loader,
            n_items=n_items,
            device=device,
            seed=args.seed,
            neg_pool_size=args.neg_pool_size,
            top_k=10,
        )
        true_hr, shuf_hr = _shuffled_target_sanity(model, val_loader, device=device, seed=args.seed, top_k=10)

        # Strict pass/fail heuristics.
        sampled_vs_full_gap = sampled_hr - full_hr
        shuffled_gap = true_hr - shuf_hr
        pass_sampled_gap = sampled_vs_full_gap <= 0.15
        pass_shuffle = shuffled_gap >= 0.05

        results["models"][label] = {
            "checkpoint": str(ckpt),
            "full_catalog": {"hr10": full_hr, "ndcg10": full_ndcg, "num_samples": n_full},
            "sampled_negative": {"hr10": sampled_hr, "ndcg10": sampled_ndcg, "num_samples": n_sampled},
            "shuffled_target_sanity": {"true_hr10": true_hr, "shuffled_hr10": shuf_hr, "gap": shuffled_gap},
            "checks": {
                "sampled_vs_full_gap_pass": pass_sampled_gap,
                "shuffled_target_gap_pass": pass_shuffle,
            },
        }

    if args.fixed_checkpoint is not None:
        if not args.fixed_checkpoint.exists():
            raise FileNotFoundError(f"Fixed checkpoint not found: {args.fixed_checkpoint}")
        evaluate_checkpoint("fixed", args.fixed_checkpoint)

    if args.learnable_checkpoint is not None:
        if not args.learnable_checkpoint.exists():
            raise FileNotFoundError(f"Learnable checkpoint not found: {args.learnable_checkpoint}")
        evaluate_checkpoint("learnable", args.learnable_checkpoint)

    global_pass = split_leakage_pass and candidate_integrity_pass
    for m in results["models"].values():
        checks = m["checks"]
        global_pass = global_pass and checks["sampled_vs_full_gap_pass"] and checks["shuffled_target_gap_pass"]

    results["overall_pass"] = bool(global_pass)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
