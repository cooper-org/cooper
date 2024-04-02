# coding: utf8
"""
Implementation of the :py:class:`ConstrainedOptimizer` class.
"""

from typing import Optional

import torch

from cooper.multipliers import ExplicitMultiplier, Multiplier
from cooper.optim.optimizer_state import CooperOptimizerState
from cooper.utils import OneOrSequence, ensure_sequence

from ... import ConstrainedMinimizationProblem
from ..types import AlternationType


class ConstrainedOptimizer:
    """
    Optimizes a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`
    given a provided :py:class:`~cooper.formulation.Formulation`.

    A ``ConstrainedOptimizer`` includes one or more
    :class:`torch.optim.Optimizer`\\s for the primal variables. It also includes
    one or more :class:`torch.optim.Optimizer`\\s for the dual variables.

    For handling unconstrained problems in a consistent way, we provide an
    :py:class:`~cooper.optim.UnconstrainedOptimizer`. Please refer to the documentation
    of the :py:class:`~cooper.problem.ConstrainedMinimizationProblem` and
    :py:class:`~cooper.formulation.Formulation` classes for further details on
    handling unconstrained problems.

    TODO: Document that we need the multiplier to be able to do projection steps
    # on the dual variables. This is because the projection step is implemented at the
    # multiplier level and there is not enough information in the dual optimizer to
    # carry out the projection step.

    Args:
        primal_optimizers: Optimizer(s) for the primal variables (e.g. the weights of
            a model). The primal parameters can be partitioned into multiple optimizers,
            in this case ``primal_optimizers`` accepts a list of
            ``torch.optim.Optimizer``\\s.

        dual_optimizers: Optimizer(s) for the dual variables (e.g. the Lagrange
            multipliers associated with the constraints). An iterable of
            ``torch.optim.Optimizer``\\s can be passed to handle the case of several
            ``~cooper.constraints.Constraint``\\s. If dealing with an unconstrained
            problem, please use a
            :py:class:`~cooper.optim.cooper_optimizer.UnconstrainedOptimizer` instead.

        multipliers: Multiplier(s) associated with the constrained optimization problem.
            We keep a reference to the multipliers to post-process them after the dual
            optimizer steps.

    """

    extrapolation: bool
    alternation_type: AlternationType

    def __init__(
        self,
        primal_optimizers: OneOrSequence[torch.optim.Optimizer],
        dual_optimizers: OneOrSequence[torch.optim.Optimizer],
        cmp: ConstrainedMinimizationProblem,
        multipliers: Optional[OneOrSequence[Multiplier]] = None,
    ):
        self.primal_optimizers = ensure_sequence(primal_optimizers)
        self.dual_optimizers = ensure_sequence(dual_optimizers)
        self.cmp = cmp
        self.multipliers = ensure_sequence(multipliers)

    def base_sanity_checks(self):
        """
        Perform sanity checks on the initialization of ``ConstrainedOptimizer``.
        """

        if self.primal_optimizers is None:
            raise RuntimeError("No primal optimizer(s) was provided for building a ConstrainedOptimizer.")
        if self.dual_optimizers is None:
            raise RuntimeError("No dual optimizer(s) was provided for building a ConstrainedOptimizer.")
        for dual_optimizer in self.dual_optimizers:
            for param_group in dual_optimizer.param_groups:
                if not param_group["maximize"]:
                    raise ValueError("Dual optimizers must be set to carry out maximization steps.")

    def zero_grad(self):
        """
        Sets the gradients of all optimized :py:class:`~torch.nn.parameter.Parameter`\\s
        to zero. This includes both the primal and dual variables.
        """
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.zero_grad()

        for dual_optimizer in self.dual_optimizers:
            dual_optimizer.zero_grad()

    def dual_step(self, call_extrapolation: bool = False):
        """
        Perform a gradient step on the parameters associated with the dual variables.
        Since the dual problem involves *maximizing* over the dual variables, we require
        dual optimizers which satisfy `maximize=True`.

        After being updated by the dual optimizer steps, the multipliers are
        post-processed (e.g. to ensure non-negativity for inequality constraints).

        Args:
            call_extrapolation: Whether to call ``dual_optimizer.extrapolation()`` as
                opposed to ``dual_optimizer.step()``. This is only relevant for
                :py:class:`~cooper.optim.constrained_optimizers.ExtrapolationConstrainedOptimizer`
                and should be left to ``False`` for other ``ConstrainedOptimizer``\\s.
        """

        if call_extrapolation:
            if not all([hasattr(dual_optimizer, "extrapolation") for dual_optimizer in self.dual_optimizers]):
                raise ValueError("All dual optimizers must implement an `extrapolation` method.")
            call_method = "extrapolation"
        else:
            call_method = "step"

        # Update multipliers based on current constraint violations (gradients)
        # For unobserved constraints the gradient is None, so this is a no-op.
        for dual_optimizer in self.dual_optimizers:
            getattr(dual_optimizer, call_method)()  # type: ignore

        for multiplier in self.multipliers:
            if isinstance(multiplier, ExplicitMultiplier):
                # `post_step_` is a no-op for multiplier with `enforce_positive=False`
                multiplier.post_step_()

    def state_dict(self) -> CooperOptimizerState:
        """
        Returns the state of the ConstrainedOptimizer. See
        :py:class:`~cooper.optim.constrained_optimizers.cooper_optimizer.CooperOptimizerState`.
        """

        primal_optimizer_states = [_.state_dict() for _ in self.primal_optimizers]
        dual_optimizer_states = [_.state_dict() for _ in self.dual_optimizers]
        multiplier_states = [_.state_dict() for _ in self.multipliers]

        return CooperOptimizerState(
            primal_optimizer_states=primal_optimizer_states,
            dual_optimizer_states=dual_optimizer_states,
            multiplier_states=multiplier_states,
            extrapolation=self.extrapolation,
            alternation_type=self.alternation_type,
        )
