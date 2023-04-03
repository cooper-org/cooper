# coding: utf8
"""
Implementation of the :py:class:`ExtrapolationConstrainedOptimizer` class.
"""

from typing import Callable, List, Optional, Union

import torch

from cooper.cmp import CMPState
from cooper.constraints import ConstraintGroup
from cooper.multipliers import MULTIPLIER_TYPE, ExplicitMultiplier
from cooper.utils import OneOrSequence

from .constrained_optimizer import ConstrainedOptimizer


class ExtrapolationConstrainedOptimizer(ConstrainedOptimizer):
    extrapolation = True
    alternating = False

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

    def dual_step(self, call_extrapolation: bool = False):
        """
        Perform a gradient step on the parameters associated with the dual variables.
        Since the dual problem involves *maximizing* over the dual variables, we flip
        the sign of the gradient to perform ascent.

        After being updated by the dual optimizer steps, the multipliers are
        post-processed (e.g. to ensure feasibility for equality constraints, or to
        apply dual restarts).

        Args:
            call_extrapolation: Whether to call ``dual_optimizer.extrapolation()`` as
                opposed to ``dual_optimizer.step()``. This is only relevant for
                :py:class:`~cooper.optim.constrained_optimizers.ExtrapolationConstrainedOptimizer`
                and should be left to ``False`` for other ``ConstrainedOptimizer``\\s.
        """
        for multiplier in self.multipliers:
            for param in multiplier.parameters():
                if param.grad is not None:
                    # Flip gradients for multipliers to perform ascent.
                    # We only do the flipping *right before* applying the optimizer
                    # step to avoid accidental double sign flips.
                    param.grad.mul_(-1.0)

        for dual_optimizer in self.dual_optimizers:
            # Update multipliers based on current constraint violations (gradients)
            # For unobserved constraints the gradient is None, so this is a no-op.
            if call_extrapolation:
                dual_optimizer.extrapolation()
            else:
                dual_optimizer.step()

        for multiplier in self.multipliers:
            if isinstance(multiplier, ExplicitMultiplier):
                # Select the indices of multipliers corresponding to feasible inequality constraints
                if multiplier.implicit_constraint_type == "ineq" and multiplier.weight.grad is not None:
                    # Feasibility is attained when the violation is negative. Given that
                    # the gradient sign is flipped, a negative violation corresponds to
                    # a positive gradient.
                    feasible_indices = multiplier.weight.grad > 0.0

                    # TODO(juan43ramirez): Document https://github.com/cooper-org/cooper/issues/28
                    # about the pitfalls of using dual_restars with stateful optimizers.

                    multiplier.post_step_(feasible_indices)
