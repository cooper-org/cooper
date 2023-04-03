# coding: utf8
"""
Implementation of the :py:class:`SimultaneousConstrainedOptimizer` class.
"""

from typing import List, Optional, Union

import torch

from cooper.constraints import ConstraintGroup
from cooper.multipliers import MULTIPLIER_TYPE, ExplicitMultiplier
from cooper.utils import OneOrSequence

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
        primal_optimizers: OneOrSequence[torch.optim.Optimizer],
        dual_optimizers: OneOrSequence[torch.optim.Optimizer],
        multipliers: Optional[OneOrSequence[MULTIPLIER_TYPE]] = None,
        constraint_groups: Optional[OneOrSequence[ConstraintGroup]] = None,
    ):
        super().__init__(primal_optimizers, dual_optimizers, multipliers, constraint_groups)

        self.base_sanity_checks()

    def step(self):
        """Performs a single optimization step on both the primal and dual variables."""

        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.step()

        self.dual_step()
