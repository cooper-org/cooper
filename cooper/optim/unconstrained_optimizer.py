# coding: utf8
"""
Implementation of the :py:class:`UnconstrainedOptimizer` class.
"""
import torch

from cooper.cmp import CMPState, LagrangianStore
from cooper.optim.optimizer import CooperOptimizer
from cooper.optim.types import AlternationType


class UnconstrainedOptimizer(CooperOptimizer):
    """Wraps a (sequence of) ``torch.optim.Optimizer``\\s to enable handling
    unconstrained problems in a way that is consistent with the
    :py:class:`~cooper.optim.ConstrainedOptimizer`\\s.

    Args:
        cmp: The constrained minimization problem to optimize.
        primal_optimizers: Optimizer(s) for the primal variables (e.g. the weights of
            a model). The primal parameters can be partitioned into multiple optimizers,
            in this case ``primal_optimizers`` accepts a sequence of
            ``torch.optim.Optimizer``\\s.
    """

    extrapolation = False
    alternation_type = AlternationType.FALSE

    def roll(
        self, compute_cmp_state_kwargs: dict = {}
    ) -> tuple[CMPState, torch.Tensor, LagrangianStore, LagrangianStore]:
        """Evaluates the objective function and performs a gradient update on the
        parameters.

        Args:
            compute_cmp_state_kwargs: Keyword arguments to pass to the ``compute_cmp_state`` method.
            Since this is an unconstrained optimizer, the CMPState will just contain the loss.
        """

        self.zero_grad()
        cmp_state = self.cmp.compute_cmp_state(**compute_cmp_state_kwargs)
        lagrangian_store = cmp_state.compute_primal_lagrangian()
        # The dual lagrangian store is empty for unconstrained problems
        dual_lagrangian_store = LagrangianStore()
        lagrangian_store.backward()

        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.step()

        return cmp_state, cmp_state.loss, lagrangian_store, dual_lagrangian_store
