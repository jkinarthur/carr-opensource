import math

import torch

from carr_v2.losses import UnifiedCompressionLoss, UnifiedLossConfig
from carr_v2.relaxations import soft_postcompression_weights, soft_precompression_reference
from carr_v2.selectors import GumbelLayerSelector
from carr_v2.weighting import AdaptiveLossWeights


def _simulate_dataset_signals(num_layers: int, sparsity: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    layer_ids = torch.arange(num_layers, dtype=torch.float32)
    # High sparsity -> faster collapse and weaker evidence survival.
    r_base = 0.55 - (0.015 + 0.02 * sparsity) * layer_ids
    s_base = 0.22 - (0.008 + 0.02 * sparsity) * layer_ids

    r_noise = 0.03 * torch.randn(num_layers, generator=g)
    s_noise = 0.02 * torch.randn(num_layers, generator=g)

    r_layers = torch.clamp(r_base + r_noise, min=0.02)
    evidence_layers = torch.clamp(s_base + s_noise, min=0.0)
    return r_layers, evidence_layers


def _fixed_threshold_k(r_layers: torch.Tensor, evidence_layers: torch.Tensor, eps_r: float, eps_s: float) -> int:
    for i in range(r_layers.numel()):
        if (r_layers[i] < eps_r) or (evidence_layers[i] < eps_s):
            return i
    return r_layers.numel() - 1


def run_benchmark() -> None:
    torch.manual_seed(17)

    num_layers = 12
    seq_len = 200
    layer_weights = torch.linspace(1.0, 2.0, num_layers)

    datasets = [
        ("ML-1M", 0.15),
        ("Beauty", 0.70),
        ("Toys", 0.50),
        ("Yelp", 0.60),
        ("Steam", 0.35),
    ]

    print("Dataset Benchmark: Adaptive-Lambda vs Fixed-Threshold Baseline")
    print("-" * 84)
    print(f"{'Dataset':<12}{'Policy':<20}{'Total':>10}{'Geo':>10}{'Evd':>10}{'Eff':>10}{'kappa':>10}")

    for idx, (name, sparsity) in enumerate(datasets, start=1):
        r_layers, evidence_layers = _simulate_dataset_signals(num_layers, sparsity=sparsity, seed=100 + idx)

        # Policy A: learned lambda + soft depth selector (single-step snapshot).
        selector = GumbelLayerSelector(num_layers=num_layers, init_temperature=1.2)
        adaptive_weights = AdaptiveLossWeights()
        loss_fn = UnifiedCompressionLoss(UnifiedLossConfig())

        selection = selector(training=False)
        kappa_soft = selection.soft_index
        r_pre = soft_precompression_reference(r_layers, kappa=kappa_soft)
        post_mask = soft_postcompression_weights(num_layers=num_layers, kappa=kappa_soft)
        m_star = torch.tensor(32.0)

        out_a = loss_fn(
            rec_loss=torch.tensor(0.32),
            r_layers=r_layers,
            r_pre_compress=r_pre,
            post_mask=post_mask,
            layer_weights=layer_weights,
            evidence_layers=evidence_layers,
            kappa=kappa_soft,
            m_star=m_star,
            selector_entropy=selection.entropy,
            num_layers=num_layers,
            seq_len=seq_len,
            lambda_overrides=adaptive_weights(),
        )

        # Policy B: fixed thresholds approximating classic monitor-based compression.
        k_fixed = _fixed_threshold_k(r_layers, evidence_layers, eps_r=0.10, eps_s=0.05)
        kappa_fixed = torch.tensor(float(k_fixed))
        r_pre_fixed = r_layers[max(k_fixed - 1, 0)]
        post_mask_fixed = torch.zeros(num_layers)
        post_mask_fixed[k_fixed:] = 1.0

        out_b = loss_fn(
            rec_loss=torch.tensor(0.32),
            r_layers=r_layers,
            r_pre_compress=r_pre_fixed,
            post_mask=post_mask_fixed,
            layer_weights=layer_weights,
            evidence_layers=evidence_layers,
            kappa=kappa_fixed,
            m_star=torch.tensor(32.0),
            selector_entropy=torch.tensor(0.0),
            num_layers=num_layers,
            seq_len=seq_len,
        )

        print(
            f"{name:<12}{'Adaptive-Lambda':<20}{float(out_a['total'].detach()):>10.4f}{float(out_a['geometry'].detach()):>10.4f}{float(out_a['evidence'].detach()):>10.4f}{float(out_a['efficiency'].detach()):>10.4f}{float(kappa_soft.detach()):>10.2f}"
        )
        print(
            f"{name:<12}{'Fixed-Threshold':<20}{float(out_b['total'].detach()):>10.4f}{float(out_b['geometry'].detach()):>10.4f}{float(out_b['evidence'].detach()):>10.4f}{float(out_b['efficiency'].detach()):>10.4f}{float(k_fixed):>10.2f}"
        )

    print("-" * 84)
    print("Interpretation: lower total indicates better trade-off under the selected objective.")


if __name__ == "__main__":
    run_benchmark()
