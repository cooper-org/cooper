"""
Implementation of the :py:class:`SimultaneousOptimizer` class.
"""

from cooper.optim.constrained_optimizers.constrained_optimizer import ConstrainedOptimizer
from cooper.optim.optimizer import RollOut


class SimultaneousOptimizer(ConstrainedOptimizer):
    """Optimizes a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`
    by performing simultaneous gradient updates to the primal and dual variables.
    """

    def roll(self, compute_cmp_state_kwargs: dict = None) -> RollOut:
        """Evaluates the CMPState and performs a simultaneous step on the primal and
        dual variables.

        Args:
            compute_cmp_state_kwargs: Keyword arguments to pass to the ``compute_cmp_state`` method.
        """

        if compute_cmp_state_kwargs is None:
            compute_cmp_state_kwargs = {}
        self.zero_grad()

        cmp_state = self.cmp.compute_cmp_state(**compute_cmp_state_kwargs)

        # TODO (gallego-posada): The current design goes over the constraints twice. We
        # could reduce overhead by writing a dedicated compute_lagrangian function for
        # the simultaneous updates
        primal_lagrangian_store = cmp_state.compute_primal_lagrangian()
        dual_lagrangian_store = cmp_state.compute_dual_lagrangian()

        # The order of the following operations is not important
        # because the primal and dual lagrangians have independent gradients
        primal_lagrangian_store.backward()
        dual_lagrangian_store.backward()

        # The order of the following operations is not important too
        # because they update independent primal and dual parameters
        self.primal_step()
        self.dual_step()

        return RollOut(cmp_state.loss, cmp_state, primal_lagrangian_store, dual_lagrangian_store)
