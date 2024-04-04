# coding: utf8
"""
Implementation of the :py:class:`UnconstrainedOptimizer` class.
"""

import torch

from cooper.cmp import CMPState, ConstrainedMinimizationProblem, LagrangianStore
from cooper.utils import OneOrSequence, ensure_sequence

from .constrained_optimizers.constrained_optimizer import CooperOptimizerState
from .types import AlternationType


class UnconstrainedOptimizer:
    """Wraps a (sequence of) ``torch.optim.Optimizer``\\s to enable handling
    unconstrained problems in a way that is consistent with the
    :py:class:`~cooper.optim.ConstrainedOptimizer`\\s.

    Args:
        primal_optimizers: Optimizer(s) for the primal variables (e.g. the weights of
            a model). The primal parameters can be partitioned into multiple optimizers,
            in this case ``primal_optimizers`` accepts a sequence of
            ``torch.optim.Optimizer``\\s.
    """

    extrapolation = False
    alternation_type = AlternationType.FALSE

    def __init__(self, primal_optimizers: OneOrSequence[torch.optim.Optimizer], cmp: ConstrainedMinimizationProblem):
        self.primal_optimizers = ensure_sequence(primal_optimizers)
        self.cmp = cmp

    def zero_grad(self):
        """Zero out the gradients of the primal variables."""
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.zero_grad()

    def step(self):
        """Perform a single optimization step on all primal optimizers."""
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.step()

    def roll(self, compute_cmp_state_kwargs: dict = {}) -> tuple[CMPState, LagrangianStore, LagrangianStore]:
        """Evaluates the objective function and performs a gradient update on the
        parameters.

        Args:
            compute_cmp_state_kwargs: Keyword arguments to pass to the ``compute_cmp_state`` method.
            Since this is an unconstrained optimizer, the CMPState will just contain the loss.
        """

        self.zero_grad()
        cmp_state = self.cmp.compute_cmp_state(**compute_cmp_state_kwargs)
        lagrangian_store = self.cmp.compute_primal_lagrangian(cmp_state)
        dual_lagrangian_store = LagrangianStore()
        # For unconstrained problems, the Lagrangian simply corresponds to the loss
        # loss = lagrangian_store.lagrangian
        # loss.backward()
        # TODO(merajhashemi): The previous two lines do not call lagrangian_purge. I am not sure if it is necessary.
        lagrangian_store.backward()
        self.step()

        return cmp_state, lagrangian_store, dual_lagrangian_store

    def state_dict(self) -> CooperOptimizerState:
        """Collects the state dicts of the primal optimizers and wraps them in a
        :py:class:`~cooper.optim.CooperOptimizerState` object.
        """

        primal_optimizer_states = [_.state_dict() for _ in self.primal_optimizers]
        return CooperOptimizerState(
            primal_optimizer_states=primal_optimizer_states,
            extrapolation=self.extrapolation,
            alternation_type=self.alternation_type,
        )
