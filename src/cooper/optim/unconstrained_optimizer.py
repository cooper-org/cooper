"""Implementation of the :py:class:`UnconstrainedOptimizer` class."""

from typing import Optional

from cooper.cmp import LagrangianStore
from cooper.optim.optimizer import CooperOptimizer, RollOut


class UnconstrainedOptimizer(CooperOptimizer):
    r"""Wraps a (sequence of) :py:class:`torch.optim.Optimizer`\s to enable handling
    unconstrained minimization problems in a way that is consistent with
    :py:class:`~cooper.optim.constrained_optimizers.ConstrainedOptimizer`\s.

    Args:
        cmp: The constrained minimization problem to be optimized. Providing the CMP
            as an argument for the constructor allows the optimizer to call the
            :py:meth:`~cooper.ConstrainedMinimizationProblem.compute_cmp_state`
            method within the :py:meth:`~cooper.optim.cooper_optimizer.CooperOptimizer.roll`
            method.
        primal_optimizers: Optimizer(s) for the primal variables (e.g. the weights of
            a model). The primal parameters can be partitioned into multiple optimizers,
            in this case ``primal_optimizers`` accepts a list of
            :py:class:`torch.optim.Optimizer`\s.
    """

    def roll(self, compute_cmp_state_kwargs: Optional[dict] = None) -> RollOut:
        """Evaluates the objective function and performs a gradient update on the
        parameters.

        Args:
            compute_cmp_state_kwargs: Keyword arguments to pass to the
                :py:meth:`~cooper.ConstrainedMinimizationProblem.compute_cmp_state()`
                method. Since this is an unconstrained optimizer, the CMPState will just
                contain the loss.
        """
        if compute_cmp_state_kwargs is None:
            compute_cmp_state_kwargs = {}
        self.zero_grad()
        cmp_state = self.cmp.compute_cmp_state(**compute_cmp_state_kwargs)
        lagrangian_store = cmp_state.compute_primal_lagrangian()
        lagrangian_store.backward()
        self.primal_step()

        # The dual lagrangian store is empty for unconstrained problems
        dual_lagrangian_store = LagrangianStore()

        return RollOut(cmp_state.loss, cmp_state, lagrangian_store, dual_lagrangian_store)
