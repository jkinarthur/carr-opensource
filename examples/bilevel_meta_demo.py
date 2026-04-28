import os

import torch

from carr_v2.convergence import CriticalDepthConvergenceLogger
from carr_v2.losses import UnifiedCompressionLoss, UnifiedLossConfig
from carr_v2.meta import BilevelLambdaOptimizer
from carr_v2.relaxations import soft_postcompression_weights, soft_precompression_reference
from carr_v2.selectors import GumbelLayerSelector
from carr_v2.weighting import AdaptiveLossWeights


def _build_loss(
    rec_bias: float,
    selector: GumbelLayerSelector,
    loss_fn: UnifiedCompressionLoss,
    num_layers: int,
    seq_len: int,
    lambda_overrides: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    # Simulated train/val statistics: real run should use model activations.
    r_layers = torch.rand(num_layers, requires_grad=True)
    evidence_layers = torch.rand(num_layers, requires_grad=True)
    layer_weights = torch.linspace(1.0, 2.0, num_layers)
    rec_loss = torch.tensor(rec_bias + 0.03 * torch.rand(1).item(), requires_grad=True)

    selection = selector(training=True)
    kappa = selection.soft_index
    r_pre = soft_precompression_reference(r_layers, kappa=kappa, bandwidth=1.2)
    post_mask = soft_postcompression_weights(num_layers=num_layers, kappa=kappa, sharpness=8.0)

    m_min, m_max = 8.0, 64.0
    m_star = m_min + (m_max - m_min) * torch.sigmoid(torch.tensor(0.2, requires_grad=True))

    out = loss_fn(
        rec_loss=rec_loss,
        r_layers=r_layers,
        r_pre_compress=r_pre,
        post_mask=post_mask,
        layer_weights=layer_weights,
        evidence_layers=evidence_layers,
        kappa=kappa,
        m_star=m_star,
        selector_entropy=selection.entropy,
        num_layers=num_layers,
        seq_len=seq_len,
        lambda_overrides=lambda_overrides,
    )
    return out, selection.soft_index, selection.hard_index


def run_demo() -> None:
    torch.manual_seed(11)

    num_layers = 12
    seq_len = 200

    selector = GumbelLayerSelector(num_layers=num_layers, init_temperature=2.0)
    adaptive_weights = AdaptiveLossWeights()
    loss_fn = UnifiedCompressionLoss(UnifiedLossConfig())

    inner_optimizer = torch.optim.Adam(selector.parameters(), lr=5e-2)
    lambda_optimizer = torch.optim.Adam(adaptive_weights.parameters(), lr=2e-2)
    bilevel = BilevelLambdaOptimizer(inner_optimizer=inner_optimizer, lambda_optimizer=lambda_optimizer)

    logger = CriticalDepthConvergenceLogger()

    # Simulated theorem proxy: critical depth estimated from theorem-3 style boundary.
    critical_depth_proxy = 4.5

    for epoch in range(1, 31):
        def train_closure(lmb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            out, _, _ = _build_loss(0.28, selector, loss_fn, num_layers, seq_len, lmb)
            return out

        def val_closure(lmb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            out, _, _ = _build_loss(0.30, selector, loss_fn, num_layers, seq_len, lmb)
            return out

        out = bilevel.step(train_closure, val_closure, lambda_provider=adaptive_weights)

        selection_eval = selector(training=False)
        logger.update(
            epoch=epoch,
            soft_depth=float(selection_eval.soft_index.detach().cpu()),
            hard_depth=int(selection_eval.hard_index.detach().cpu()),
            critical_depth=critical_depth_proxy,
        )

        selector.set_temperature(2.0 * (0.96 ** epoch))
        adaptive_weights.set_temperature(1.0 * (0.99 ** epoch))

        if epoch in (1, 10, 20, 30):
            print(
                f"Epoch {epoch:>2} | train={out.train_total:.4f} val={out.val_total:.4f} "
                f"| lambdas=({out.lambda_geometry:.3f}, {out.lambda_evidence:.3f}, {out.lambda_efficiency:.3f}, {out.lambda_entropy:.3f})"
            )

    out_dir = os.path.join("outputs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "depth_convergence.csv")
    logger.to_csv(csv_path)

    print("\nConvergence summary:")
    print(f"  mean |kappa-k*| (last 10): {logger.mean_gap_last(window=10):.3f}")
    print(f"  stabilized@0.25: {logger.stabilized(window=10, tol=0.25)}")
    print(f"  wrote: {csv_path}")


if __name__ == "__main__":
    run_demo()
