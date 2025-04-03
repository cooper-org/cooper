"""Implementation of the :py:class:`ConstrainedOptimizer` class."""

import abc
from typing import Any

import torch

from cooper.cmp import ConstrainedMinimizationProblem
from cooper.optim.optimizer import CooperOptimizer, RollOut
from cooper.utils import OneOrSequence


class ConstrainedOptimizer(CooperOptimizer, abc.ABC):
    r"""Optimizes a :py:class:`~cooper.ConstrainedMinimizationProblem`.

    A :py:class:`ConstrainedOptimizer` includes one or more
    :class:`torch.optim.Optimizer`\s for the primal variables. It also includes
    one or more :class:`torch.optim.Optimizer`\s for the dual variables.

    For handling unconstrained problems in a consistent way, we provide the
    :py:class:`~cooper.optim.UnconstrainedOptimizer` class.

    Args:
        cmp: The constrained minimization problem to be optimized. Providing the CMP
            as an argument for the constructor allows the optimizer to call the
            :py:meth:`~cooper.ConstrainedMinimizationProblem.compute_cmp_state`
            method within the :py:meth:`~cooper.optim.cooper_optimizer.CooperOptimizer.roll`
            method. Additionally, in the case of a constrained optimizer, the CMP
            enables access to the multipliers'
            :py:meth:`~cooper.multipliers.Multiplier.post_step_` method which must be
            called after the multiplier update.
        primal_optimizers: Optimizer(s) for the primal variables (e.g. the weights of
            a model). The primal parameters can be partitioned into multiple optimizers,
            in this case ``primal_optimizers`` accepts a list of
            :py:class:`torch.optim.Optimizer`\s.
        dual_optimizers: Optimizer(s) for the dual variables (e.g. the Lagrange
            multipliers associated with the constraints). A sequence of
            :py:class:`torch.optim.Optimizer`\s can be passed to handle the case of
            several :py:class:`~cooper.constraints.Constraint`\s. If dealing with an
            unconstrained problem, please use an
            :py:class:`~cooper.optim.UnconstrainedOptimizer` instead.

    """

    def __init__(
        self,
        cmp: ConstrainedMinimizationProblem,
        primal_optimizers: OneOrSequence[torch.optim.Optimizer],
        dual_optimizers: OneOrSequence[torch.optim.Optimizer],
    ) -> None:
        super().__init__(cmp=cmp, primal_optimizers=primal_optimizers, dual_optimizers=dual_optimizers)
        self.base_sanity_checks()
        # custom_sanity_checks are implemented in the derived classes
        self.custom_sanity_checks()

    def base_sanity_checks(self) -> None:
        """Performs sanity checks on the initialization of ``ConstrainedOptimizer``.

        Raises:
            TypeError: If no primal or dual optimizers are provided.
            ValueError: If any dual optimizer is not configured with `maximize=True`.
        """
        if self.primal_optimizers is None:
            raise TypeError("No primal optimizer(s) was provided for building a ConstrainedOptimizer.")
        if self.dual_optimizers is None:
            raise TypeError("No dual optimizer(s) was provided for building a ConstrainedOptimizer.")
        for dual_optimizer in self.dual_optimizers:
            for param_group in dual_optimizer.param_groups:
                if not param_group["maximize"]:
                    raise ValueError("Dual optimizers must be set to carry out maximization steps.")

    def custom_sanity_checks(self) -> None:
        """Performs custom sanity checks on the initialization of ``ConstrainedOptimizer``."""

    @torch.no_grad()
    def dual_step(self) -> None:
        """Performs a gradient step on the parameters associated with the dual variables.
        Since the dual problem involves *maximizing* over the dual variables, we require
        dual optimizers which satisfy ``maximize=True``.

        After being updated by the dual optimizer steps, the multipliers are
        post-processed (e.g. to ensure non-negativity for inequality constraints).
        """
        # Update multipliers based on current constraint violations (gradients)
        # For unobserved constraints the gradient is None, so this is a no-op.
        for dual_optimizer in self.dual_optimizers:
            dual_optimizer.step()

        # post steps include, among other things, ensuring that
        # multipliers for inequality constraints are non-negative.
        for multiplier in self.cmp.multipliers():
            multiplier.post_step_()

    @abc.abstractmethod
    def roll(self, *args: Any, **kwargs: Any) -> RollOut:
        """Performs a full update step on the primal and dual variables."""
