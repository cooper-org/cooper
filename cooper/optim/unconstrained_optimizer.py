# coding: utf8

from typing import Callable

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


class UnconstrainedExtrapolationOptimizer:
    """Wraps a (sequence of) ``cooper.optim.ExtragradientOptimizer``\\s to enable
    handling unconstrained problems in a way that is consistent with the
    :py:class:`~cooper.optim.ConstrainedOptimizer`\\s.

    This class handles the calls to ``primal_optimizer.extrapolation()`` and
    ``primal_optimizer.step()`` internally in the ``step`` method.

    Args:
        primal_optimizers: Optimizer(s) for the primal variables (e.g. the weights of
            a model). The primal parameters can be partitioned into multiple optimizers,
            in this case ``primal_optimizers`` accepts a sequence of
            ``torch.optim.Optimizer``\\s.
    """

    extrapolation = True
    alternating = False

    def __init__(self, primal_optimizers: OneOrSequence[torch.optim.Optimizer]):
        self.primal_optimizers = ensure_sequence(primal_optimizers)
        self.sanity_checks()

    def sanity_checks(self):
        """
        Perform sanity checks on the initialization of
        ``UnconstrainedExtrapolationOptimizer``.

        Raises:
            RuntimeError: Tried to construct an ExtrapolationConstrainedOptimizer but
                some of the provided optimizers do not have an extrapolation method.
        """

        are_primal_extra_optims = [hasattr(_, "extrapolation") for _ in self.primal_optimizers]

        if not all(are_primal_extra_optims):
            raise RuntimeError(
                """Some of the provided optimizers do not have an extrapolation method.
                Please ensure that all optimizers are extrapolation capable."""
            )

    def zero_grad(self):
        """Zero out the gradients of the primal variables."""
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.zero_grad()

    def step(self, compute_cmp_state_fn: Callable[..., CMPState]):
        """Performs an extrapolation step, followed by an update step.
        ``compute_cmp_state_fn`` is used to populate gradients on the extrapolated point

        Args:
            compute_cmp_state_fn: ``Callable`` for re-evaluating the objective and
                constraints when performing alternating updates. Defaults to None.
        """
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.extrapolation()  # type: ignore

        self.zero_grad()

        # `compute_cmp_state_fn` is re-evaluated at the extrapolated point since the
        # state of the primal parameters has been updated.
        cmp_state_after_extrapolation = compute_cmp_state_fn()
        _ = cmp_state_after_extrapolation.populate_lagrangian()

        # Populate gradients at extrapolation point
        cmp_state_after_extrapolation.backward()

        # After this, the calls to `step` will update the stored copies of the
        # parameters with the newly computed gradients.
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.step()

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
