from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LayerSelectionOutput:
    probs: torch.Tensor
    soft_index: torch.Tensor
    hard_index: torch.Tensor
    entropy: torch.Tensor


class GumbelLayerSelector(nn.Module):
    """Differentiable layer selector using Gumbel-Softmax."""

    def __init__(self, num_layers: int, init_temperature: float = 2.0):
        super().__init__()
        if num_layers <= 1:
            raise ValueError("num_layers must be > 1")
        self.num_layers = num_layers
        self.logits = nn.Parameter(torch.zeros(num_layers))
        self.temperature = init_temperature

    def set_temperature(self, temperature: float) -> None:
        self.temperature = max(float(temperature), 1e-3)

    def forward(self, training: bool = True) -> LayerSelectionOutput:
        if training:
            probs = F.gumbel_softmax(self.logits, tau=self.temperature, hard=False, dim=-1)
        else:
            probs = torch.softmax(self.logits, dim=-1)

        layer_ids = torch.arange(self.num_layers, dtype=probs.dtype, device=probs.device)
        soft_index = (probs * layer_ids).sum()
        hard_index = probs.argmax(dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum()

        return LayerSelectionOutput(
            probs=probs,
            soft_index=soft_index,
            hard_index=hard_index,
            entropy=entropy,
        )
