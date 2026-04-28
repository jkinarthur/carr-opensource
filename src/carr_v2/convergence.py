from __future__ import annotations

from dataclasses import dataclass, field

import csv


@dataclass
class ConvergenceRecord:
    epoch: int
    soft_depth: float
    hard_depth: int
    critical_depth: float
    abs_gap: float


@dataclass
class CriticalDepthConvergenceLogger:
    """Tracks convergence of learned depth toward critical depth proxy."""

    records: list[ConvergenceRecord] = field(default_factory=list)

    def update(self, epoch: int, soft_depth: float, hard_depth: int, critical_depth: float) -> None:
        self.records.append(
            ConvergenceRecord(
                epoch=epoch,
                soft_depth=soft_depth,
                hard_depth=hard_depth,
                critical_depth=critical_depth,
                abs_gap=abs(soft_depth - critical_depth),
            )
        )

    def mean_gap_last(self, window: int = 10) -> float:
        tail = self.records[-window:] if len(self.records) >= window else self.records
        if not tail:
            return 0.0
        return sum(r.abs_gap for r in tail) / len(tail)

    def stabilized(self, window: int = 10, tol: float = 0.25) -> bool:
        return self.mean_gap_last(window=window) <= tol

    def to_csv(self, file_path: str) -> None:
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "soft_depth", "hard_depth", "critical_depth", "abs_gap"])
            for r in self.records:
                writer.writerow([r.epoch, f"{r.soft_depth:.6f}", r.hard_depth, f"{r.critical_depth:.6f}", f"{r.abs_gap:.6f}"])
