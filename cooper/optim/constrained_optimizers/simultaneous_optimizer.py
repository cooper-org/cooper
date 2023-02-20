# coding: utf8
"""
Implementation of the :py:class:`SimultaneousConstrainedOptimizer` class.
"""

from typing import Callable, List, Optional, Union

import torch

from cooper.cmp import CMPState
from cooper.constraints import ConstraintGroup

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
        """

        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.step()

        self.dual_step()

    def dual_step(self):

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

                _multiplier.post_step(feasible_indices, restart_value=0.0)
