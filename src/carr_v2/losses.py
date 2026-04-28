from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .metrics import normalized_efficiency_cost


@dataclass
class UnifiedLossConfig:
    lambda_geometry: float = 0.2
    lambda_evidence: float = 0.2
    lambda_efficiency: float = 0.05
    lambda_entropy: float = 0.01
    epsilon_s: float = 0.05


class UnifiedCompressionLoss(nn.Module):
    """
    Unified objective for CARR-v2:
    L = L_rec + lambda1 * L_geometry + lambda2 * L_evidence + lambda3 * L_eff + lambda4 * L_entropy
    """

    def __init__(self, config: UnifiedLossConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        rec_loss: torch.Tensor,
        r_layers: torch.Tensor,
        r_pre_compress: torch.Tensor,
        post_mask: torch.Tensor,
        layer_weights: torch.Tensor,
        evidence_layers: torch.Tensor,
        kappa: torch.Tensor,
        m_star: torch.Tensor,
        selector_entropy: torch.Tensor,
        num_layers: int,
        seq_len: int,
        lambda_overrides: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if r_layers.ndim != 1:
            raise ValueError("r_layers must be 1D")
        if layer_weights.shape != r_layers.shape:
            raise ValueError("layer_weights must match r_layers shape")
        if post_mask.shape != r_layers.shape:
            raise ValueError("post_mask must match r_layers shape")

        weighted_post = post_mask * layer_weights
        weighted_post = weighted_post / (weighted_post.sum() + 1e-8)

        geometry_loss = (weighted_post * (r_layers - r_pre_compress).pow(2)).sum()
        evidence_hinge = torch.clamp(self.config.epsilon_s - evidence_layers, min=0.0).pow(2)
        evidence_loss = (weighted_post * evidence_hinge).sum()
        efficiency_loss = normalized_efficiency_cost(kappa, m_star, num_layers=num_layers, seq_len=seq_len)
        entropy_commit_loss = selector_entropy

        if lambda_overrides is None:
            lambda_geometry = torch.as_tensor(self.config.lambda_geometry, device=rec_loss.device, dtype=rec_loss.dtype)
            lambda_evidence = torch.as_tensor(self.config.lambda_evidence, device=rec_loss.device, dtype=rec_loss.dtype)
            lambda_efficiency = torch.as_tensor(self.config.lambda_efficiency, device=rec_loss.device, dtype=rec_loss.dtype)
            lambda_entropy = torch.as_tensor(self.config.lambda_entropy, device=rec_loss.device, dtype=rec_loss.dtype)
        else:
            lambda_geometry = lambda_overrides["geometry"]
            lambda_evidence = lambda_overrides["evidence"]
            lambda_efficiency = lambda_overrides["efficiency"]
            lambda_entropy = lambda_overrides["entropy"]

        total = (
            rec_loss
            + lambda_geometry * geometry_loss
            + lambda_evidence * evidence_loss
            + lambda_efficiency * efficiency_loss
            + lambda_entropy * entropy_commit_loss
        )

        return {
            "total": total,
            "rec": rec_loss,
            "geometry": geometry_loss,
            "evidence": evidence_loss,
            "efficiency": efficiency_loss,
            "entropy": entropy_commit_loss,
            "lambda_geometry": lambda_geometry,
            "lambda_evidence": lambda_evidence,
            "lambda_efficiency": lambda_efficiency,
            "lambda_entropy": lambda_entropy,
        }
