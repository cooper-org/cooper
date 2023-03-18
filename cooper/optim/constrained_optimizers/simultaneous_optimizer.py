# coding: utf8
"""
Implementation of the :py:class:`SimultaneousConstrainedOptimizer` class.
"""

from typing import List, Optional, Union

import torch

from cooper.constraints import ConstraintGroup
from cooper.multipliers import MULTIPLIER_TYPE, ExplicitMultiplier

from .constrained_optimizer import ConstrainedOptimizer


class SimultaneousConstrainedOptimizer(ConstrainedOptimizer):
    """
    Optimizes a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`
    by performing simultaneous gradient updates to the primal and dual variables.

    A ``SimultaneousConstrainedOptimizer`` includes one or more
    :class:`torch.optim.Optimizer`\\s for the primal variables. It also includes
    one or more :class:`torch.optim.Optimizer`\\s for the dual variables.

    Args:
        primal_optimizers: Optimizer(s) for the primal variables (e.g. the weights of
            a model). The primal parameters can be partitioned into multiple optimizers,
            in this case ``primal_optimizers`` accepts a list of
            ``torch.optim.Optimizer``\\s.

        dual_optimizer: Optimizer(s) for the dual variables (e.g. the Lagrange
            multipliers associated with the constraints). An iterable of
            ``torch.optim.Optimizer``\\s can be passed to handle the case of several
            ``~cooper.constraints.ConstraintGroup``\\s. If dealing with an unconstrained
            problem, please use a
            :py:class:`~cooper.optim.cooper_optimizer.UnconstrainedOptimizer` instead.

    """

    extrapolation = False
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

    def step(self):
        """Performs a single optimization step on both the primal and dual variables."""

        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.step()

        self.dual_step()

    def dual_step(self):
        """
        Perform a gradient step on the parameters associated with the dual variables.
        Since the dual problem involves *maximizing* over the dual variables, we flip
        the sign of the gradient to perform ascent.

        After being updated by the dual optimizer steps, the multipliers are
        post-processed (e.g. to ensure feasibility for equality constraints, or to
        apply dual restarts).
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
