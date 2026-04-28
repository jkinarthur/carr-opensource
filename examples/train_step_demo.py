import torch

from carr_v2.losses import UnifiedCompressionLoss, UnifiedLossConfig
from carr_v2.relaxations import soft_postcompression_weights, soft_precompression_reference
from carr_v2.selectors import GumbelLayerSelector
from carr_v2.tracking import DepthTracker
from carr_v2.weighting import AdaptiveLossWeights


def demo_train_step() -> None:
    torch.manual_seed(7)

    num_layers = 12
    seq_len = 200

    selector = GumbelLayerSelector(num_layers=num_layers, init_temperature=2.0)
    adaptive_weights = AdaptiveLossWeights()
    loss_fn = UnifiedCompressionLoss(UnifiedLossConfig())
    tracker = DepthTracker()
    optimizer = torch.optim.Adam(
        list(selector.parameters()) + list(adaptive_weights.parameters()),
        lr=5e-2,
    )

    for epoch in range(1, 21):
        optimizer.zero_grad()

        # Dummy recommendation loss from your ranking model.
        rec_loss = torch.tensor(0.30 + 0.002 * epoch, requires_grad=True)

        # Full-layer statistics; in real training these come from model activations.
        r_layers = torch.rand(num_layers, requires_grad=True)
        layer_weights = torch.linspace(1.0, 2.0, num_layers)
        evidence_layers = torch.rand(num_layers, requires_grad=True)

        selection = selector(training=True)
        kappa = selection.soft_index

        # Differentiable approximation of R(kappa-) from layerwise scores.
        r_pre = soft_precompression_reference(r_layers, kappa=kappa, bandwidth=1.2)
        post_mask = soft_postcompression_weights(num_layers=num_layers, kappa=kappa, sharpness=8.0)

        # Continuous relaxed register width m* in [m_min, m_max].
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
            lambda_overrides=adaptive_weights(),
        )

        out["total"].backward()
        optimizer.step()

        # Mild annealing to encourage eventual hard layer commitment.
        selector.set_temperature(2.0 * (0.96 ** epoch))
        adaptive_weights.set_temperature(1.0 * (0.99 ** epoch))
        tracker.update(selection.soft_index, selection.hard_index)

        if epoch in (1, 5, 10, 20):
            print(f"Epoch {epoch:>2}: hard={int(selection.hard_index.item())}, soft={float(kappa.detach()):.3f}, temp={selector.temperature:.3f}")

    print("\nFinal loss breakdown:")
    for k, v in out.items():
        print(f"  {k:>18}: {float(v.detach()):.6f}")

    print("\nDepth dynamics summary:")
    print("  Mean soft depth (last 10):", f"{tracker.mean_soft_last(window=10):.3f}")
    print("  Mode hard depth (last 10):", tracker.hard_mode_last(window=10))


if __name__ == "__main__":
    demo_train_step()
