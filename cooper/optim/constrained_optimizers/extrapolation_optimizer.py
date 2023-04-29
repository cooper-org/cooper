# coding: utf8
"""
Implementation of the :py:class:`ExtrapolationConstrainedOptimizer` class.
"""

from typing import Callable, Optional

import torch

from cooper.cmp import CMPState, LagrangianStore
from cooper.constraints import ConstraintGroup
from cooper.multipliers import MULTIPLIER_TYPE
from cooper.utils import OneOrSequence

from ..types import AlternatingType
from .constrained_optimizer import ConstrainedOptimizer


class ExtrapolationConstrainedOptimizer(ConstrainedOptimizer):
    """Optimizes a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`
    by performing extrapolation updates to the primal and dual variables.
    """

    # TODO(gallego-posada): Add equations to illustrate the extrapolation updates

    extrapolation = True
    alternating = AlternatingType.FALSE

    def __init__(
        self,
        primal_optimizers: OneOrSequence[torch.optim.Optimizer],
        dual_optimizers: OneOrSequence[torch.optim.Optimizer],
        multipliers: Optional[OneOrSequence[MULTIPLIER_TYPE]] = None,
        constraint_groups: Optional[OneOrSequence[ConstraintGroup]] = None,
    ):
        super().__init__(primal_optimizers, dual_optimizers, multipliers, constraint_groups)

        self.base_sanity_checks()

        self.custom_sanity_checks()

    def custom_sanity_checks(self):
        """
        Perform sanity checks on the initialization of
        ``ExtrapolationConstrainedOptimizer``.

        Raises:
            RuntimeError: Tried to construct an ExtrapolationConstrainedOptimizer but
                some of the provided optimizers do not have an extrapolation method.
        """

        are_primal_extra_optims = [hasattr(_, "extrapolation") for _ in self.primal_optimizers]
        are_dual_extra_optims = [hasattr(_, "extrapolation") for _ in self.dual_optimizers]

        if not all(are_primal_extra_optims) or not all(are_dual_extra_optims):
            raise RuntimeError(
                """Some of the provided optimizers do not have an extrapolation method.
                Please ensure that all optimizers are extrapolation capable."""
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
            self.dual_step(call_extrapolation=call_extrapolation)

    def roll(
        self, compute_cmp_state_fn: Callable[..., CMPState], return_multipliers: bool = False
    ) -> tuple[CMPState, LagrangianStore]:
        """Performs a full extrapolation step on the primal and dual variables.

        Note that the forward and backward computations associated with the CMPState
        and Lagrangian are carried out twice, since we compute an "extra" gradient.

        Args:
            compute_cmp_state_fn: ``Callable`` for evaluating the CMPState.

            return_multipliers: When `True`, we return the updated value of the
                multipliers for the observed constraints.
        """

        self.zero_grad()
        cmp_state_pre_extrapolation = compute_cmp_state_fn()
        lagrangian_store_pre_extrapolation = cmp_state_pre_extrapolation.populate_lagrangian(return_multipliers=False)
        cmp_state_pre_extrapolation.backward()
        self.step(call_extrapolation=True)

        # Perform an update step
        self.zero_grad()
        cmp_state_post_extrapolation = compute_cmp_state_fn()
        lagrangian_store_post_extrapolation = cmp_state_post_extrapolation.populate_lagrangian(
            return_multipliers=return_multipliers
        )
        cmp_state_post_extrapolation.backward()
        self.step(call_extrapolation=False)

        return cmp_state_post_extrapolation, lagrangian_store_post_extrapolation
