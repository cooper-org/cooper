# coding: utf8
"""
Implementation of the :py:class:`AlternatingConstrainedOptimizer` class.
"""

from typing import Callable, List, Optional, Union

import torch

from cooper.cmp import CMPState
from cooper.constraints import ConstraintGroup
from cooper.multipliers import MULTIPLIER_TYPE, ExplicitMultiplier

from .constrained_optimizer import ConstrainedOptimizer


class AlternatingConstrainedOptimizer(ConstrainedOptimizer):
    extrapolation = False
    alternating = True

    def __init__(
        self,
        primal_optimizers: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]],
        dual_optimizers: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]],
        multipliers: Optional[Union[MULTIPLIER_TYPE, List[MULTIPLIER_TYPE]]] = None,
        constraint_groups: Optional[Union[List[ConstraintGroup], ConstraintGroup]] = None,
    ):

        super().__init__(primal_optimizers, dual_optimizers, multipliers, constraint_groups)

        self.base_sanity_checks()

    def step(
        self,
        compute_cmp_state_fn: Optional[Callable[..., CMPState]] = None,
        compute_violations_fn: Optional[Callable[..., CMPState]] = None,
        return_multipliers: bool = False,
    ):
        """Performs an alternating optimization step: use the existing gradients to
        update the primal variables, and re-evaluate the constraints (or full CMP state)
        to update the dual variables.

        Args:
            compute_cmp_state_fn: ``Callable`` for re-evaluating the objective and
                constraints when performing alternating updates. Defaults to None.

            compute_violations_fn: ``Callable`` for re-evaluating the constraint
                violations only when performing alternating updates. When this argument
                is provided, it takes precedence over the `compute_cmp_state_fn`.
                Defaults to None.
        """

        if (compute_cmp_state_fn is None) and (compute_violations_fn is None):
            error_message = "One of `compute_cmp_state_fn` or `compute_violations_fn` required for alternating update."
            raise RuntimeError(error_message)

        # Start by performing a gradient step on the primal variables
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.step()

        # Zero-out gradients for dual variables since they were already populated.
        for dual_optimizer in self.dual_optimizers:
            dual_optimizer.zero_grad()

        with torch.no_grad():
            # We skip gradient wrt primal parameters to avoid wasteful computation,
            # since we only need the gradient wrt the dual variables.
            # Note that the dual variables do not intervene in the compuation of the
            # CMP state.
            if compute_violations_fn is not None:
                cmp_state_after_primal_update = compute_violations_fn()
            else:
                cmp_state_after_primal_update = compute_cmp_state_fn()

            if cmp_state_after_primal_update._dual_lagrangian is not None:
                error_message = (
                    "Expected a fresh CMP state for alternating update but the provided state has a non-None value"
                    " for the `_dual_lagrangian` attribute."
                )
                raise RuntimeError(error_message)

        # We have already computed the new CMP state with the new values of the
        # parameters. Now we only need to recalculate the Lagrangian so we can get the
        # gradients wrt the multipliers.
        #
        # Note that the call to defect_fn might _not_ have populated the loss. This is
        # not a problem since we only need to compute the gradient wrt the dual
        # variables.
        lagrangian_return = cmp_state_after_primal_update.populate_lagrangian(return_multipliers=return_multipliers)

        # We only need to compute the gradients for the dual variables, so we skip
        # the primal_backward call.
        cmp_state_after_primal_update.dual_backward()

        self.dual_step()

        if return_multipliers:
            multipliers = lagrangian_return[1]
            return cmp_state_after_primal_update, multipliers
        else:
            return cmp_state_after_primal_update
