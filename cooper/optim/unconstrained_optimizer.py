import warnings
from typing import List, Union

import torch

from cooper.formulation import Formulation, UnconstrainedFormulation

from .constrained_optimizers.cooper_optimizer import (
    CooperOptimizer,
    CooperOptimizerState,
)


class UnconstrainedOptimizer(CooperOptimizer):
    """
    Fallback class to handle unconstrained problems in a unified way.

    Args:
        cmp: ``ConstrainedMinimizationProblem`` we aim to solve and which gives
            rise to the Lagrangian.
        ineq_init: Initialization values for the inequality multipliers.
        eq_init: Initialization values for the equality multipliers.

    """

    def __init__(
        self,
        formulation: UnconstrainedFormulation,
        primal_optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer],
    ):

        if not isinstance(formulation, UnconstrainedFormulation):
            warnings.warn(
                """Creating an unconstrained optimizer for a constrained formulation. \
                The optimizer will ignore the constraints of this CMP."""
            )

        self.formulation = formulation
        self.cmp = self.formulation.cmp

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
        return CooperOptimizerState(primal_optimizer_states=primal_optimizer_states)
