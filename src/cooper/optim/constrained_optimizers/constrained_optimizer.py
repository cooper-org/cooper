"""Implementation of the :py:class:`ConstrainedOptimizer` class."""

import torch

from cooper.cmp import ConstrainedMinimizationProblem
from cooper.optim.optimizer import CooperOptimizer
from cooper.utils import OneOrSequence


class ConstrainedOptimizer(CooperOptimizer):
    r"""Optimizes a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`
    given a provided :py:class:`~cooper.formulation.Formulation`.

    A ``ConstrainedOptimizer`` includes one or more
    :class:`torch.optim.Optimizer`\s for the primal variables. It also includes
    one or more :class:`torch.optim.Optimizer`\s for the dual variables.

    For handling unconstrained problems in a consistent way, we provide an
    :py:class:`~cooper.optim.UnconstrainedOptimizer`. Please refer to the documentation
    of the :py:class:`~cooper.problem.ConstrainedMinimizationProblem` and
    :py:class:`~cooper.formulation.Formulation` classes for further details on
    handling unconstrained problems.

    Args:
        primal_optimizers: Optimizer(s) for the primal variables (e.g. the weights of
            a model). The primal parameters can be partitioned into multiple optimizers,
            in this case ``primal_optimizers`` accepts a list of
            ``torch.optim.Optimizer``\s.

        dual_optimizers: Optimizer(s) for the dual variables (e.g. the Lagrange
            multipliers associated with the constraints). An iterable of
            ``torch.optim.Optimizer``\s can be passed to handle the case of several
            ``~cooper.constraints.Constraint``\s. If dealing with an unconstrained
            problem, please use a
            :py:class:`~cooper.optim.cooper_optimizer.UnconstrainedOptimizer` instead.

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
        """Perform sanity checks on the initialization of ``ConstrainedOptimizer``."""
        if self.primal_optimizers is None:
            raise TypeError("No primal optimizer(s) was provided for building a ConstrainedOptimizer.")
        if self.dual_optimizers is None:
            raise TypeError("No dual optimizer(s) was provided for building a ConstrainedOptimizer.")
        for dual_optimizer in self.dual_optimizers:
            for param_group in dual_optimizer.param_groups:
                if not param_group["maximize"]:
                    raise ValueError("Dual optimizers must be set to carry out maximization steps.")

    def custom_sanity_checks(self) -> None:
        """Perform custom sanity checks on the initialization of ``ConstrainedOptimizer``."""

    @torch.no_grad()
    def dual_step(self) -> None:
        """Perform a gradient step on the parameters associated with the dual variables.
        Since the dual problem involves *maximizing* over the dual variables, we require
        dual optimizers which satisfy `maximize=True`.

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
