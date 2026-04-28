from __future__ import annotations

import torch


def soft_precompression_reference(
    r_per_layer: torch.Tensor,
    kappa: torch.Tensor,
    bandwidth: float = 1.2,
) -> torch.Tensor:
    """
    Differentiable approximation of R(kappa-) using a Gaussian kernel centered at (kappa - 0.5).

    Args:
        r_per_layer: Shape [L], collapse scores per layer.
        kappa: Continuous compression depth in [0, L-1].
        bandwidth: Smoothing width for the kernel.
    """
    if r_per_layer.ndim != 1:
        raise ValueError("r_per_layer must be 1D")

    layer_ids = torch.arange(r_per_layer.numel(), device=r_per_layer.device, dtype=r_per_layer.dtype)
    center = kappa - 0.5
    logits = -((layer_ids - center) ** 2) / (2.0 * (bandwidth**2))
    weights = torch.softmax(logits, dim=0)
    return (weights * r_per_layer).sum()


def soft_postcompression_weights(
    num_layers: int,
    kappa: torch.Tensor,
    sharpness: float = 8.0,
) -> torch.Tensor:
    """Differentiable mask approximating 1[l >= kappa] for post-compression layers."""
    layer_ids = torch.arange(num_layers, device=kappa.device, dtype=kappa.dtype)
    return torch.sigmoid((layer_ids - kappa) * sharpness)
