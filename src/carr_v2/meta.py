from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


LossClosure = Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]
LambdaProvider = Callable[[], dict[str, torch.Tensor]]


@dataclass
class BilevelStepOutput:
    train_total: float
    val_total: float
    lambda_geometry: float
    lambda_evidence: float
    lambda_efficiency: float
    lambda_entropy: float


class BilevelLambdaOptimizer:
    """
    First-order bilevel optimizer for adaptive lambda learning.

    Inner step: update model/selector on train loss.
    Outer step: update lambda parameters on validation loss.
    """

    def __init__(self, inner_optimizer: torch.optim.Optimizer, lambda_optimizer: torch.optim.Optimizer):
        self.inner_optimizer = inner_optimizer
        self.lambda_optimizer = lambda_optimizer

    def step(
        self,
        train_loss_closure: LossClosure,
        val_loss_closure: LossClosure,
        lambda_provider: LambdaProvider,
    ) -> BilevelStepOutput:
        self.inner_optimizer.zero_grad(set_to_none=True)
        train_out = train_loss_closure(lambda_provider())
        train_out["total"].backward()
        self.inner_optimizer.step()

        self.lambda_optimizer.zero_grad(set_to_none=True)
        val_out = val_loss_closure(lambda_provider())
        val_out["total"].backward()
        self.lambda_optimizer.step()

        current_lambdas = lambda_provider()
        return BilevelStepOutput(
            train_total=float(train_out["total"].detach().cpu()),
            val_total=float(val_out["total"].detach().cpu()),
            lambda_geometry=float(current_lambdas["geometry"].detach().cpu()),
            lambda_evidence=float(current_lambdas["evidence"].detach().cpu()),
            lambda_efficiency=float(current_lambdas["efficiency"].detach().cpu()),
            lambda_entropy=float(current_lambdas["entropy"].detach().cpu()),
        )
