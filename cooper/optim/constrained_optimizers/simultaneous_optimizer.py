# coding: utf8
"""
Implementation of the :py:class:`SimultaneousConstrainedOptimizer` class.
"""

from typing import Callable, List, Optional, Union

import torch

from cooper.formulation import Formulation
from cooper.problem import CMPState

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

    def __init__(
        self,
        formulation: Formulation,
        primal_optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer],
        dual_optimizer: Optional[torch.optim.Optimizer] = None,
        dual_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        dual_restarts: bool = False,
    ):
        self.formulation = formulation

        if isinstance(primal_optimizers, torch.optim.Optimizer):
            self.primal_optimizers = [primal_optimizers]
        else:
            self.primal_optimizers = primal_optimizers

        self.dual_optimizer = dual_optimizer
        self.dual_scheduler = dual_scheduler

        self.dual_restarts = dual_restarts

        self.extrapolation = False
        self.alternating = False

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

        # If necessary, instantiate dual components
        if not hasattr(self.dual_optimizer, "param_groups"):
            self.instantiate_dual_optimizer_and_scheduler()

        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.step()

        self.dual_step()

    def dual_step(self):

        # Flip gradients for multipliers to perform ascent.
        # We only do the flipping *right before* applying the optimizer step to
        # avoid accidental double sign flips.
        for multiplier in self.formulation.state():
            if multiplier is not None:
                multiplier.grad.mul_(-1.0)

        # Update multipliers based on current constraint violations (gradients)
        self.dual_optimizer.step()

        if self.formulation.ineq_multipliers is not None:
            if self.dual_restarts:
                # "Reset" value of inequality multipliers to zero as soon as
                # solution becomes feasible
                self.restart_dual_variables()

            # Apply projection step to inequality multipliers
            self.formulation.ineq_multipliers.project_()
