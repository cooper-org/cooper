# coding: utf8
"""
Implementation of the :py:class:`SimultaneousConstrainedOptimizer` class.
"""

from typing import List, Union

import torch

from cooper.constraints import ConstraintGroup
from cooper.multipliers import ExplicitMultiplier

from .constrained_optimizer import ConstrainedOptimizer


class SimultaneousConstrainedOptimizer(ConstrainedOptimizer):
    """
    Optimizes a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`
    given a provided :py:class:`~cooper.formulation.Formulation`.

    A ``ConstrainedOptimizer`` includes one or more
    :class:`torch.optim.Optimizer`\\s for the primal variables. It also includes
    a :class:`torch.optim.Optimizer` for the dual variables associated with the
    provided ``Formulation``.

    Args:
        formulation: ``Formulation`` of the ``ConstrainedMinimizationProblem``
            to be optimized.

        primal_optimizers: Fully instantiated ``torch.optim.Optimizer``\\s used
            to optimize the primal parameters (e.g. model parameters). The primal
            parameters can be partitioned into multiple optimizers, in this case
            ``primal_optimizers`` accepts a list of ``torch.optim.Optimizer``\\s.

        dual_optimizer: Partially instantiated ``torch.optim.Optimizer``
            used to optimize the dual variables (e.g. Lagrange multipliers).

        dual_scheduler: Partially instantiated
            ``torch.optim.lr_scheduler._LRScheduler`` used to schedule the
            learning rate of the dual variables. Defaults to None.

        dual_restarts: If True, perform "restarts" on the Lagrange
            multipliers associated with inequality constraints: whenever the
            constraint is satisfied, directly set the multiplier to zero.
            Defaults to False.

    """

    extrapolation = False
    alternating = False

    def __init__(
        self,
        constraint_groups: Union[List[ConstraintGroup], ConstraintGroup],
        primal_optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer],
        dual_optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer],
    ):
        super().__init__(constraint_groups, primal_optimizers, dual_optimizers)

        self.base_sanity_checks()

    def step(self):
        """Performs a single optimization step on both the primal and dual variables."""

        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.step()

        self.dual_step()

    def dual_step(self):
        # FIXME(juan43ramirez): do not couple the constraint groups with the
        # dual_optimizers. Ensure that sanity_checks allow for different number of each

        for constraint in self.constraint_groups:
            for param in constraint.multiplier.parameters():
                if param.grad is not None:
                    # Flip gradients for multipliers to perform ascent.
                    # We only do the flipping *right before* applying the optimizer step to
                    # avoid accidental double sign flips.
                    param.grad.mul_(-1.0)

        for dual_optimizer in self.dual_optimizers:
            # Update multipliers based on current constraint violations (gradients)
            # For unobserved constraints the gradient is None, so this is a no-op.
            dual_optimizer.step()

        for constraint in self.constraint_groups:
            multiplier = constraint.multiplier

            if isinstance(multiplier, ExplicitMultiplier):
                if multiplier.weight.grad is not None:
                    feasible_indices = None
                    if multiplier.implicit_constraint_type == "ineq":
                        # Select the indices of multipliers corresponding to feasible constraints

                        # Feasibility is attained when the violation is negative. Given that
                        # the gradient sign is flipped, a negative violation corresponds to
                        # a positive gradient.
                        feasible_indices = multiplier.weight.grad > 0.0

                        # TODO(juan43ramirez): Document https://github.com/cooper-org/cooper/issues/28
                        # about the pitfalls of using dual_restars with stateful optimizers.

                        multiplier.post_step_(feasible_indices, restart_value=0.0)
