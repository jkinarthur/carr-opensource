"""
CARR-v2 Production Trainer
==========================
End-to-end bilevel training loop using the full-scale CARRBackbone (d_model=256,
12 transformer layers) and publication-grade dataset sizes.

Features:
  - CUDA auto-detection with mixed-precision (torch.amp)
  - DataParallel for multi-GPU EC2 instances
  - AdamW + linear-warmup / cosine-annealing LR schedule
  - Gradient clipping (norm=1.0)
  - Bilevel λ meta-update once per epoch on a fresh validation batch
  - Checkpoint save every --ckpt_every epochs + best-model tracking
  - Resume from checkpoint via --resume
  - Real dataset support via --data_path (TSV: user_id/item_id/timestamp)

Outputs (in --out_dir, default outputs/):
  checkpoints/ckpt_epoch{N:04d}.pt
  checkpoints/best_model.pt
  trainer_depth_convergence.csv
  trainer_lambdas.csv

Usage on EC2:
  python examples/trainer.py --n_epochs 200 --n_users 100000 --n_items 20000
  python examples/trainer.py --data_path /data/ml-1m.tsv --n_epochs 200
"""

import argparse
import csv
import math
import os

import torch
import torch.nn.functional as F

from carr_v2.backbone import CARRBackbone, _compute_collapse_score
from carr_v2.convergence import CriticalDepthConvergenceLogger
from carr_v2.data import make_loaders
from carr_v2.losses import UnifiedCompressionLoss, UnifiedLossConfig
from carr_v2.meta import BilevelLambdaOptimizer
from carr_v2.relaxations import soft_postcompression_weights, soft_precompression_reference
from carr_v2.selectors import GumbelLayerSelector
from carr_v2.weighting import AdaptiveLossWeights


def _layer_diagnostics(
    hiddens: list[torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert per-layer hidden states to R(l) and evidence proxy tensors."""
    r_vals, s_vals = [], []
    for h in hiddens:
        r, s = _compute_collapse_score(h, n_intents=8)
        r_vals.append(r)
        s_vals.append(s)
    return (
        torch.tensor(r_vals, dtype=torch.float32, device=device),
        torch.tensor(s_vals, dtype=torch.float32, device=device),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CARR-v2 Production Trainer")
    p.add_argument("--n_epochs",     type=int,   default=200)
    p.add_argument("--n_users",      type=int,   default=100_000)
    p.add_argument("--n_items",      type=int,   default=20_000)
    p.add_argument("--n_layers",     type=int,   default=12)
    p.add_argument("--d_model",      type=int,   default=256)
    p.add_argument("--n_heads",      type=int,   default=8)
    p.add_argument("--seq_len",      type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=512)
    p.add_argument("--lr_model",     type=float, default=3e-4)
    p.add_argument("--lr_lambda",    type=float, default=1e-2)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--n_warmup",     type=int,   default=10,
                   help="Linear LR warmup epochs before cosine decay")
    p.add_argument("--ckpt_every",   type=int,   default=20)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--out_dir",      type=str,   default="outputs")
    p.add_argument("--data_path",    type=str,   default=None,
                   help="Path to real interaction TSV (user_id/item_id/timestamp). "
                        "Omit to use synthetic data.")
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--resume",       type=str,   default=None,
                   help="Checkpoint .pt to resume from")
    return p


def train(
    n_epochs: int = 200,
    n_users: int = 100_000,
    n_items: int = 20_000,
    n_layers: int = 12,
    d_model: int = 256,
    n_heads: int = 8,
    seq_len: int = 50,
    batch_size: int = 512,
    lr_model: float = 3e-4,
    lr_lambda: float = 1e-2,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    critical_depth_proxy: float = 4.5,
    n_warmup: int = 10,
    ckpt_every: int = 20,
    seed: int = 42,
    out_dir: str = "outputs",
    data_path: str | None = None,
    num_workers: int = 4,
    resume: str | None = None,
) -> None:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"Device: {device}  |  AMP: {use_amp}  |  Epochs: {n_epochs}")

    train_loader, val_loader, n_items = make_loaders(
        n_users=n_users, n_items=n_items, seq_len=seq_len,
        batch_size=batch_size, num_workers=num_workers, seed=seed,
        data_path=data_path,
    )
    print(f"Dataset: {len(train_loader.dataset)} train / {len(val_loader.dataset)} val "
          f"| items: {n_items}")

    model = CARRBackbone(
        num_items=n_items, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, max_seq=seq_len + 10,
    ).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"DataParallel: {torch.cuda.device_count()} GPUs")

    selector = GumbelLayerSelector(num_layers=n_layers, init_temperature=2.0).to(device)
    adaptive_weights = AdaptiveLossWeights().to(device)
    loss_fn = UnifiedCompressionLoss(UnifiedLossConfig()).to(device)
    layer_weights = torch.linspace(1.0, 2.0, n_layers, device=device)
    m_star = torch.tensor(float(n_layers) * 0.667, device=device)

    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    inner_params = list(raw_model.parameters()) + list(selector.parameters())
    inner_optimizer = torch.optim.AdamW(inner_params, lr=lr_model, weight_decay=weight_decay)
    lambda_optimizer = torch.optim.Adam(adaptive_weights.parameters(), lr=lr_lambda)
    bilevel = BilevelLambdaOptimizer(inner_optimizer=inner_optimizer, lambda_optimizer=lambda_optimizer)
    convergence_log = CriticalDepthConvergenceLogger()

    def _lr_schedule(epoch: int) -> float:
        if epoch < n_warmup:
            return float(epoch + 1) / float(n_warmup)
        progress = (epoch - n_warmup) / max(n_epochs - n_warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(inner_optimizer, lr_lambda=_lr_schedule)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    start_epoch = 1
    best_val_loss = float("inf")

    if resume is not None:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        selector.load_state_dict(ckpt["selector"])
        adaptive_weights.load_state_dict(ckpt["adaptive_weights"])
        inner_optimizer.load_state_dict(ckpt["inner_optimizer"])
        lambda_optimizer.load_state_dict(ckpt["lambda_optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        print(f"Resumed from {resume} at epoch {start_epoch}")

    depth_csv_path = os.path.join(out_dir, "trainer_depth_convergence.csv")
    lambda_csv_path = os.path.join(out_dir, "trainer_lambdas.csv")
    with open(depth_csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "soft_depth", "hard_depth", "critical_depth", "abs_gap"])
    with open(lambda_csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "epoch", "lam_geometry", "lam_evidence", "lam_efficiency", "lam_entropy",
            "train_loss", "val_loss", "lr",
        ])

    for epoch in range(start_epoch, n_epochs + 1):
        # ---- Inner loop: train on all batches with current λ ----
        raw_model.train()
        epoch_train_loss = 0.0
        n_batches = 0
        last_seqs_t = last_targets_t = None

        with torch.no_grad():
            current_lmb = adaptive_weights()

        for seqs, targets in train_loader:
            seqs = seqs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            inner_optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits, hiddens = model(seqs)
                rec_loss = F.cross_entropy(logits, targets)
                r_layers, evidence_layers = _layer_diagnostics(hiddens, device)
                selection = selector(training=True)
                kappa = selection.soft_index
                r_pre = soft_precompression_reference(r_layers, kappa=kappa)
                post_mask = soft_postcompression_weights(num_layers=n_layers, kappa=kappa)
                out = loss_fn(
                    rec_loss=rec_loss,
                    r_layers=r_layers, r_pre_compress=r_pre,
                    post_mask=post_mask, layer_weights=layer_weights,
                    evidence_layers=evidence_layers, kappa=kappa,
                    m_star=m_star, selector_entropy=selection.entropy,
                    num_layers=n_layers, seq_len=seq_len,
                    lambda_overrides=current_lmb,
                )

            scaler.scale(out["total"]).backward()
            scaler.unscale_(inner_optimizer)
            torch.nn.utils.clip_grad_norm_(inner_params, max_norm=grad_clip)
            scaler.step(inner_optimizer)
            scaler.update()

            epoch_train_loss += float(out["total"].detach())
            n_batches += 1
            last_seqs_t, last_targets_t = seqs, targets

        epoch_train_loss /= max(n_batches, 1)

        # ---- Bilevel λ meta-update: one step on last train batch + first val batch ----
        seqs_v, targets_v = next(iter(val_loader))
        seqs_v = seqs_v.to(device)
        targets_v = targets_v.to(device)
        seqs_t = last_seqs_t
        targets_t = last_targets_t

        def train_closure(lmb: dict) -> dict:
            raw_model.train()
            logits_t, hid_t = raw_model(seqs_t)
            rl, el = _layer_diagnostics(hid_t, device)
            sel = selector(training=True)
            k = sel.soft_index
            return loss_fn(
                rec_loss=F.cross_entropy(logits_t, targets_t),
                r_layers=rl, r_pre_compress=soft_precompression_reference(rl, k),
                post_mask=soft_postcompression_weights(n_layers, k),
                layer_weights=layer_weights, evidence_layers=el, kappa=k,
                m_star=m_star, selector_entropy=sel.entropy,
                num_layers=n_layers, seq_len=seq_len, lambda_overrides=lmb,
            )

        def val_closure(lmb: dict) -> dict:
            raw_model.eval()
            with torch.no_grad():
                logits_v, hid_v = raw_model(seqs_v)
            rl_v, el_v = _layer_diagnostics(hid_v, device)
            sel_v = selector(training=False)
            k_v = sel_v.soft_index
            return loss_fn(
                rec_loss=F.cross_entropy(logits_v, targets_v),
                r_layers=rl_v, r_pre_compress=soft_precompression_reference(rl_v, k_v),
                post_mask=soft_postcompression_weights(n_layers, k_v),
                layer_weights=layer_weights, evidence_layers=el_v, kappa=k_v,
                m_star=m_star, selector_entropy=sel_v.entropy,
                num_layers=n_layers, seq_len=seq_len, lambda_overrides=lmb,
            )

        bilevel_result = bilevel.step(train_closure, val_closure, lambda_provider=adaptive_weights)
        val_loss = float(bilevel_result.val_total)

        # ---- LR + temperature annealing ----
        scheduler.step()
        selector.set_temperature(max(0.2, 2.0 * (0.97 ** epoch)))
        adaptive_weights.set_temperature(max(0.1, 1.0 * (0.995 ** epoch)))

        # ---- Convergence logging ----
        with torch.no_grad():
            sel_eval = selector(training=False)
        soft_d = float(sel_eval.soft_index)
        hard_d = int(sel_eval.hard_index)
        convergence_log.update(epoch=epoch, soft_depth=soft_d, hard_depth=hard_d,
                               critical_depth=critical_depth_proxy)

        with open(depth_csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, f"{soft_d:.6f}", hard_d,
                f"{critical_depth_proxy:.6f}", f"{abs(soft_d - critical_depth_proxy):.6f}",
            ])
        current_lr = inner_optimizer.param_groups[0]["lr"]
        with open(lambda_csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch,
                f"{bilevel_result.lambda_geometry:.6f}",
                f"{bilevel_result.lambda_evidence:.6f}",
                f"{bilevel_result.lambda_efficiency:.6f}",
                f"{bilevel_result.lambda_entropy:.6f}",
                f"{epoch_train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{current_lr:.2e}",
            ])

        # ---- Checkpointing ----
        if epoch % ckpt_every == 0 or epoch == n_epochs:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "model": raw_model.state_dict(),
                "selector": selector.state_dict(),
                "adaptive_weights": adaptive_weights.state_dict(),
                "inner_optimizer": inner_optimizer.state_dict(),
                "lambda_optimizer": lambda_optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }, ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(raw_model.state_dict(), os.path.join(ckpt_dir, "best_model.pt"))

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:>4}/{n_epochs} | "
                f"train={epoch_train_loss:.4f}  val={val_loss:.4f} | "
                f"κ={soft_d:.2f} (k={hard_d}) | "
                f"λ=({bilevel_result.lambda_geometry:.3f}, {bilevel_result.lambda_evidence:.3f}, "
                f"{bilevel_result.lambda_efficiency:.3f}, {bilevel_result.lambda_entropy:.3f}) | "
                f"lr={current_lr:.2e}"
            )

    print(f"\nTraining complete.")
    print(f"Convergence stabilized@0.5: {convergence_log.stabilized(window=10, tol=0.5)}")
    print(f"Mean |κ−k*| last 10 epochs: {convergence_log.mean_gap_last(10):.3f}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Outputs in: {out_dir}/")


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train(**vars(args))
