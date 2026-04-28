from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class AdaptiveWeightsConfig:
    """Configuration for simplex-constrained adaptive loss weights."""

    init_geometry: float = 0.35
    init_evidence: float = 0.30
    init_efficiency: float = 0.20
    init_entropy: float = 0.15
    temperature: float = 1.0


class AdaptiveLossWeights(nn.Module):
    """Learns lambda_1..lambda_4 on a probability simplex using softmax logits."""

    def __init__(self, config: AdaptiveWeightsConfig = AdaptiveWeightsConfig()):
        super().__init__()
        init = torch.tensor(
            [
                config.init_geometry,
                config.init_evidence,
                config.init_efficiency,
                config.init_entropy,
            ],
            dtype=torch.float32,
        )
        init = init / init.sum()
        self.logits = nn.Parameter(torch.log(init))
        self.temperature = max(float(config.temperature), 1e-3)

    def set_temperature(self, temperature: float) -> None:
        self.temperature = max(float(temperature), 1e-3)

    def forward(self) -> dict[str, torch.Tensor]:
        weights = torch.softmax(self.logits / self.temperature, dim=0)
        return {
            "geometry": weights[0],
            "evidence": weights[1],
            "efficiency": weights[2],
            "entropy": weights[3],
            "vector": weights,
        }
