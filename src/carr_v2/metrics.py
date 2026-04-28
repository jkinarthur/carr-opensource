from __future__ import annotations

import torch


def reasoning_collapse_score(within_scatter: torch.Tensor, between_scatter: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """R(l) = Tr(Sigma_W) / Tr(Sigma_B)."""
    return within_scatter / (between_scatter + eps)


def evidence_hinge_loss(evidence_values: torch.Tensor, min_survival: float) -> torch.Tensor:
    """Penalize evidence survival below epsilon_S."""
    return torch.clamp(min_survival - evidence_values, min=0.0).pow(2).mean()


def normalized_efficiency_cost(kappa: torch.Tensor, m_star: torch.Tensor, num_layers: int, seq_len: int) -> torch.Tensor:
    """Continuous relaxation of compute ratio vs full model cost."""
    l = float(num_layers)
    t = float(seq_len)
    return (kappa * t + (l - kappa) * m_star) / (l * t)
