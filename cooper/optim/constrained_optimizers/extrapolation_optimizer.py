# coding: utf8
"""
Implementation of the :py:class:`ExtrapolationConstrainedOptimizer` class.
"""

from typing import Optional

import torch

from cooper.cmp import CMPState, ConstrainedMinimizationProblem, LagrangianStore
from cooper.multipliers import Multiplier
from cooper.utils import OneOrSequence

from ..types import AlternationType
from .constrained_optimizer import ConstrainedOptimizer


class ExtrapolationConstrainedOptimizer(ConstrainedOptimizer):
    """Optimizes a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`
    by performing extrapolation updates to the primal and dual variables.
    """

    # TODO(gallego-posada): Add equations to illustrate the extrapolation updates

    extrapolation = True
    alternation_type = AlternationType.FALSE

    def __init__(
        self,
        primal_optimizers: OneOrSequence[torch.optim.Optimizer],
        dual_optimizers: OneOrSequence[torch.optim.Optimizer],
        cmp: ConstrainedMinimizationProblem,
        multipliers: Optional[OneOrSequence[Multiplier]] = None,
    ):
        super().__init__(primal_optimizers, dual_optimizers, cmp, multipliers)

        self.base_sanity_checks()

        self.custom_sanity_checks()

    def custom_sanity_checks(self):
        """
        Perform sanity checks on the initialization of
        ``ExtrapolationConstrainedOptimizer``.

        Raises:
            RuntimeError: Tried to construct an ExtrapolationConstrainedOptimizer but
                some of the provided optimizers do not have an extrapolation method.
            RuntimeError: Using an ExtrapolationConstrainedOptimizer together with
                multipliers that have ``restart_on_feasible=True`` is not supported.
        """

        are_primal_extra_optims = [hasattr(_, "extrapolation") for _ in self.primal_optimizers]
        are_dual_extra_optims = [hasattr(_, "extrapolation") for _ in self.dual_optimizers]

        if not all(are_primal_extra_optims) or not all(are_dual_extra_optims):
            raise RuntimeError(
                """Some of the provided optimizers do not have an extrapolation method.
                Please ensure that all optimizers are extrapolation capable."""
            )

        for multiplier in self.multipliers:
            if getattr(multiplier, "restart_on_feasible", False):
                raise RuntimeError(
                    """Using restart on feasible for multipliers is not supported in
                    conjunction with the ExtrapolationConstrainedOptimizer."""
                )

    def step(self, call_extrapolation: bool = False):
        """Performs an extrapolation step or update step on both the primal and dual
        variables.

        Args:
            call_extrapolation: Whether to call ``primal_optimizer.extrapolation()`` as
                opposed to ``primal_optimizer.step()``. Defaults to False.
        """

        call_method = "extrapolation" if call_extrapolation else "step"

        for primal_optimizer in self.primal_optimizers:
            getattr(primal_optimizer, call_method)()  # type: ignore

            # FIXME(gallego-posada): This line should not be indented inside the loop!
            self.dual_step(call_extrapolation=call_extrapolation)

    def roll(self, compute_cmp_state_kwargs: dict = {}) -> tuple[CMPState, LagrangianStore]:
        """Performs a full extrapolation step on the primal and dual variables.

        Note that the forward and backward computations associated with the CMPState
        and Lagrangian are carried out twice, since we compute an "extra" gradient.

        Args:
            compute_cmp_state_kwargs: Keyword arguments to pass to the ``compute_cmp_state`` method.
        """

        self.zero_grad()
        cmp_state_pre_extrapolation = self.cmp.compute_cmp_state(**compute_cmp_state_kwargs)
        lagrangian_store_pre_extrapolation = self.cmp.populate_lagrangian(cmp_state_pre_extrapolation)
        lagrangian_store_pre_extrapolation.backward()
        self.step(call_extrapolation=True)

        # Perform an update step
        self.zero_grad()
        cmp_state_post_extrapolation = self.cmp.compute_cmp_state(**compute_cmp_state_kwargs)
        lagrangian_store_post_extrapolation = self.cmp.populate_lagrangian(cmp_state_post_extrapolation)
        lagrangian_store_pre_extrapolation.backward()
        self.step(call_extrapolation=False)

        return cmp_state_post_extrapolation, lagrangian_store_post_extrapolation
