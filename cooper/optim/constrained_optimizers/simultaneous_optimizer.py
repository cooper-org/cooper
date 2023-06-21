# coding: utf8
"""
Implementation of the :py:class:`SimultaneousOptimizer` class.
"""

from typing import Callable, Optional

import torch

from cooper.cmp import CMPState, LagrangianStore
from cooper.constraints import ConstraintGroup
from cooper.multipliers import Multiplier
from cooper.utils import OneOrSequence

from ..types import AlternatingType
from .constrained_optimizer import ConstrainedOptimizer


class SimultaneousOptimizer(ConstrainedOptimizer):
    """Optimizes a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`
    by performing simultaneous gradient updates to the primal and dual variables.
    """

    extrapolation = False
    alternating = AlternatingType.FALSE

    def __init__(
        self,
        primal_optimizers: OneOrSequence[torch.optim.Optimizer],
        dual_optimizers: OneOrSequence[torch.optim.Optimizer],
        multipliers: Optional[OneOrSequence[Multiplier]] = None,
        constraint_groups: Optional[OneOrSequence[ConstraintGroup]] = None,
    ):
        super().__init__(primal_optimizers, dual_optimizers, multipliers, constraint_groups)

        self.base_sanity_checks()

    def step(self):
        """Performs a single optimization step on both the primal and dual variables."""

        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.step()

        self.dual_step(call_extrapolation=False)

    def roll(
        self, compute_cmp_state_fn: Callable[..., CMPState], return_multipliers: bool = False
    ) -> tuple[CMPState, LagrangianStore]:
        """Evaluates the CMPState and performs a simultaneous step on the primal and
        dual variables.

        Args:
            compute_cmp_state_fn: ``Callable`` for evaluating the CMPState.

            return_multipliers: When `True`, we return the updated value of the
                multipliers for the observed constraints.
        """

        self.zero_grad()
        cmp_state = compute_cmp_state_fn()
        lagrangian_store = cmp_state.populate_lagrangian(return_multipliers=return_multipliers)
        cmp_state.backward()
        self.step()

        return cmp_state, lagrangian_store
