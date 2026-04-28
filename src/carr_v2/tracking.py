from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class DepthTracker:
    """Tracks learned compression depth dynamics across training."""

    soft_depths: list[float] = field(default_factory=list)
    hard_depths: list[int] = field(default_factory=list)

    def update(self, soft_depth: torch.Tensor, hard_depth: torch.Tensor) -> None:
        self.soft_depths.append(float(soft_depth.detach().cpu()))
        self.hard_depths.append(int(hard_depth.detach().cpu()))

    def mean_soft_last(self, window: int = 10) -> float:
        vals = self.soft_depths[-window:] if len(self.soft_depths) >= window else self.soft_depths
        return float(sum(vals) / max(len(vals), 1))

    def hard_mode_last(self, window: int = 10) -> int:
        vals = self.hard_depths[-window:] if len(self.hard_depths) >= window else self.hard_depths
        if not vals:
            return -1
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        return max(counts, key=counts.get)
