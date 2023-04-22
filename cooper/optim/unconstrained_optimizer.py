# coding: utf8

import torch

from cooper.cmp import CMPState
from cooper.utils import OneOrSequence, ensure_sequence

from .constrained_optimizers.constrained_optimizer import CooperOptimizerState


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
    alternating = False

    def __init__(self, primal_optimizers: OneOrSequence[torch.optim.Optimizer]):
        self.primal_optimizers = ensure_sequence(primal_optimizers)

    def zero_grad(self):
        """Zero out the gradients of the primal variables."""
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.zero_grad()

    def step(self):
        """Perform a single optimization step on all primal optimizers."""
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.step()

    def roll(self, cmp_state: CMPState) -> torch.Tensor:
        # TODO(gallego-posada): Document this
        """Perform a single optimization step on all primal optimizers."""

        self.zero_grad()
        lagrangian_store = cmp_state.populate_lagrangian()
        # For unconstrained problems, the Lagrangian simply corresponds to the loss
        loss = lagrangian_store.lagrangian
        loss.backward()
        self.step()

        return loss

    def state_dict(self) -> CooperOptimizerState:
        """Collects the state dicts of the primal optimizers and wraps them in a
        :py:class:`~cooper.optim.CooperOptimizerState` object.
        """

        primal_optimizer_states = [_.state_dict() for _ in self.primal_optimizers]
        return CooperOptimizerState(
            primal_optimizer_states=primal_optimizer_states,
            extrapolation=self.extrapolation,
            alternating=self.alternating,
        )
