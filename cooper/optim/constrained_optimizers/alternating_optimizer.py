# coding: utf8
"""
Implementation of the :py:class:`AlternatingConstrainedOptimizer` class.
"""

from typing import Callable, List, Optional, Union

import torch

from cooper.cmp import CMPState
from cooper.constraints import ConstraintGroup

from .constrained_optimizer import ConstrainedOptimizer


class AlternatingConstrainedOptimizer(ConstrainedOptimizer):
    extrapolation = False
    alternating = True

    def __init__(
        self,
        constraint_groups: Union[List[ConstraintGroup], ConstraintGroup],
        primal_optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer],
        dual_optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer],
    ):
        super().__init__(constraint_groups, primal_optimizers, dual_optimizers)

        self.base_sanity_checks()

    def step(
        self,
        closure: Optional[Callable[..., CMPState]] = None,
        *closure_args,
        defect_fn: Optional[Callable[..., CMPState]] = None,
        **closure_kwargs,
    ):
        """
        Performs a single optimization step on both the primal and dual
        variables.

        Args:
            closure: Closure ``Callable`` required for re-evaluating the
                objective and constraints when performing alternating updates.
                Defaults to None.

            *closure_args: Arguments to be passed to the closure function
                when re-evaluating.

            **closure_kwargs: Keyword arguments to be passed to the closure
                function when re-evaluating.
        """

        if (closure is None) and (defect_fn is None):
            raise RuntimeError("At least one of closure or defect_fn must be provided for alternating updates.")

        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.step()

        alternate_cmp_state = self.populate_alternating_dual_gradient(
            closure, defect_fn, *closure_args, **closure_kwargs
        )

        self.dual_step()

        return alternate_cmp_state

    def populate_alternating_dual_gradient(
        self,
        closure: Optional[Callable[..., CMPState]] = None,
        *closure_args,
        defect_fn: Optional[Callable[..., CMPState]] = None,
        **closure_kwargs,
    ):
        # TODO(juan43ramirez): rename alternate_cmp_state to something more informative.

        # Zero-out gradients for dual variables since they were already populated. We
        # also zero-out primal gradients for safety although not really necessary.
        self.zero_grad()

        # Once having updated the primal parameters, re-compute gradient wrt
        # multipliers. Skip gradient wrt primal parameters to avoid wasteful
        # computation, as we only need gradient wrt multipliers.
        with torch.no_grad():
            assert closure is not None or defect_fn is not None

            if defect_fn is not None:
                alternate_cmp_state = defect_fn(*closure_args, **closure_kwargs)

                if alternate_cmp_state.loss is None:
                    # Store a placeholder loss which does not contribute to the Lagrangian
                    alternate_cmp_state.loss = 0.0

            elif closure is not None:
                alternate_cmp_state = closure(*closure_args, **closure_kwargs)

        # We have already computed the new CMP state with the new values of the
        # parameters. Now we only need to recalculate the Lagrangian so we can get the
        # gradients wrt the multipliers.
        #
        # Note that the call to defect_fn might _not_ have populated the loss.
        # This is not a problem since we only need to compute the gradient wrt
        # the multipliers.
        _ = alternate_cmp_state.populate_lagrangian()

        # We only want to compute the gradients for the dual variables
        alternate_cmp_state.dual_backward()

        return alternate_cmp_state

    def dual_step(self):
        # TODO(juan43ramirez): this function is exactly the same as SimultaneousOptimizer.dual_step.

        for constraint, dual_optimizer in zip(self.constraint_groups, self.dual_optimizers):
            # TODO(juan43ramirez): which convention will we use to access the gradient
            # of the multipliers?
            _multiplier = constraint.multiplier

            if _multiplier.weight.grad is not None:
                # Flip gradients for multipliers to perform ascent.
                # We only do the flipping *right before* applying the optimizer step to
                # avoid accidental double sign flips.
                _multiplier.weight.grad.mul_(-1.0)

                # Update multipliers based on current constraint violations (gradients)
                dual_optimizer.step()

                # Select the indices of multipliers corresponding to feasible constraints
                if _multiplier.implicit_constraint_type == "ineq":
                    # TODO(juan43ramirez): could alternatively access the state of the
                    # constraint, which would be more transparent.

                    # Feasibility is attained when the violation is negative. Given that
                    # the gradient sign is flipped, a negative violation corresponds to
                    # a positive gradient.
                    feasible_indices = _multiplier.weight.grad > 0.0
                else:
                    # TODO(juan43ramirez): add comment
                    feasible_indices = None

                _multiplier.post_step_(feasible_indices, restart_value=0.0)
