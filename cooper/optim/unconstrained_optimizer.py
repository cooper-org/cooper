import warnings
from typing import List, Union

import torch

from .constrained_optimizers.constrained_optimizer import CooperOptimizerState


class UnconstrainedOptimizer:
    """
    Fallback class to handle unconstrained problems in a unified way.

    Args:

    """

    extrapolation = False
    alternating = False

    def __init__(self, primal_optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer]):
        if isinstance(primal_optimizers, torch.optim.Optimizer):
            self.primal_optimizers = [primal_optimizers]
        else:
            self.primal_optimizers = primal_optimizers

    def zero_grad(self):
        """Zero out the gradients of the primal variables."""
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.zero_grad()

    def step(self):
        """Perform a single optimization step on all primal optimizers."""
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.step()

    def state_dict(self) -> CooperOptimizerState:
        """
        Collects the state dicts of the primal optimizers and returns them as a
        `CooperOptimizerState` object.
        """
        primal_optimizer_states = [_.state_dict() for _ in self.primal_optimizers]
        return CooperOptimizerState(
            primal_optimizer_states=primal_optimizer_states,
            extrapolation=self.extrapolation,
            alternating=self.alternating,
        )


# TODO(juan43ramirez): implement UnconstrainedExtrapolationOptimizer
