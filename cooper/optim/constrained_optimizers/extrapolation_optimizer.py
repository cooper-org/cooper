# coding: utf8
"""
Implementation of the :py:class:`ExtrapolationConstrainedOptimizer` class.
"""

from typing import Callable, List, Optional, Union

import torch

from cooper.cmp import CMPState
from cooper.constraints import ConstraintGroup
from cooper.multipliers import MULTIPLIER_TYPE

from .constrained_optimizer import ConstrainedOptimizer


class ExtrapolationConstrainedOptimizer(ConstrainedOptimizer):
    extrapolation = True
    alternating = False

    def __init__(
        self,
        primal_optimizers: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]],
        dual_optimizers: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]],
        multipliers: Optional[Union[MULTIPLIER_TYPE, List[MULTIPLIER_TYPE]]] = None,
        constraint_groups: Optional[Union[List[ConstraintGroup], ConstraintGroup]] = None,
    ):
        super().__init__(primal_optimizers, dual_optimizers, multipliers, constraint_groups)

        self.base_sanity_checks()

        self.custom_sanity_checks()

    def custom_sanity_checks(self):
        """
        Perform sanity checks on the initialization of ``ExtrapolationConstrainedOptimizer``.

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

    def step(self, compute_cmp_state_fn: Callable[..., CMPState]):
        """Performs an extrapolation step on both the primal and dual variables,
        followed by an update step. ``compute_cmp_state_fn`` is used to populate
        gradients after the extrapolation step.

        Args:
            compute_cmp_state_fn: ``Callable`` for re-evaluating the objective and
                constraints when performing alternating updates. Defaults to None.
        """

        if compute_cmp_state_fn is None:
            raise RuntimeError("`compute_cmp_state_fn` must be provided to step when using extrapolation.")

        # Store parameter copy and compute t+1/2 iterates
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.extrapolation()  # type: ignore

        # Call to dual_step flips sign of gradients, then triggers call to
        # dual_optimizer.extrapolation and applies post_step_.
        self.dual_step(call_extrapolation=True)

        self.zero_grad()

        # `compute_cmp_state_fn` is re-evaluated at the extrapolated point since the
        # state of the primal parameters has been updated.
        cmp_state_after_extrapolation = compute_cmp_state_fn()
        _ = cmp_state_after_extrapolation.populate_lagrangian()

        # Populate gradients at extrapolation point
        cmp_state_after_extrapolation.backward()

        # After this, the calls to `step` will update the stored copies of the
        # parameters with the newly computed gradients.
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.step()

        self.dual_step(call_extrapolation=False)
