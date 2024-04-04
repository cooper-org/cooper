# coding: utf8
"""
Implementation of the :py:class:`SimultaneousOptimizer` class.
"""

import torch

from cooper.cmp import CMPState, ConstrainedMinimizationProblem, LagrangianStore
from cooper.utils import OneOrSequence

from ..types import AlternationType
from .constrained_optimizer import ConstrainedOptimizer


class SimultaneousOptimizer(ConstrainedOptimizer):
    """Optimizes a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`
    by performing simultaneous gradient updates to the primal and dual variables.
    """

    extrapolation = False
    alternation_type = AlternationType.FALSE

    def __init__(
        self,
        primal_optimizers: OneOrSequence[torch.optim.Optimizer],
        dual_optimizers: OneOrSequence[torch.optim.Optimizer],
        cmp: ConstrainedMinimizationProblem,
    ):
        super().__init__(primal_optimizers, dual_optimizers, cmp)

        self.base_sanity_checks()

    @torch.no_grad()
    def step(self):
        """Performs a single optimization step on both the primal and dual variables."""

        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.step()

        self.dual_step(call_extrapolation=False)

    def roll(self, compute_cmp_state_kwargs: dict = {}) -> tuple[CMPState, LagrangianStore, LagrangianStore]:
        """Evaluates the CMPState and performs a simultaneous step on the primal and
        dual variables.

        Args:
            compute_cmp_state_kwargs: Keyword arguments to pass to the ``compute_cmp_state`` method.
        """

        self.zero_grad()

        cmp_state = self.cmp.compute_cmp_state(**compute_cmp_state_kwargs)

        # TODO: The current design goes over the constraints twice. We could reduce
        #  overhead by writing a dedicated compute_lagrangian function for the simultaneous updates
        primal_lagrangian_store = cmp_state.compute_primal_lagrangian()
        dual_lagrangian_store = cmp_state.compute_dual_lagrangian()

        # The order of the following operations is not important
        primal_lagrangian_store.backward()
        dual_lagrangian_store.backward()

        self.step()

        return cmp_state, primal_lagrangian_store, dual_lagrangian_store
