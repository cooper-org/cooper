"""
Implementation of the :py:class:`SimultaneousOptimizer` class.
"""
from cooper.optim.constrained_optimizers.constrained_optimizer import ConstrainedOptimizer
from cooper.optim.optimizer import RollOut
from cooper.optim.types import AlternationType


class SimultaneousOptimizer(ConstrainedOptimizer):
    """Optimizes a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`
    by performing simultaneous gradient updates to the primal and dual variables.
    """

    extrapolation = False
    alternation_type = AlternationType.FALSE

    def roll(self, compute_cmp_state_kwargs: dict = {}) -> RollOut:
        """Evaluates the CMPState and performs a simultaneous step on the primal and
        dual variables.

        Args:
            compute_cmp_state_kwargs: Keyword arguments to pass to the ``compute_cmp_state`` method.
        """

        self.zero_grad()

        cmp_state = self.cmp.compute_cmp_state(**compute_cmp_state_kwargs)

        # TODO: The current design goes over the constraints twice. We could reduce
        #  overhead by writing a dedicated compute_lagrangian function for the simultaneous updates
        primal_lagrangian_store = cmp_state.compute_primal_lagrangian()
        dual_lagrangian_store = cmp_state.compute_dual_lagrangian()

        # The order of the following operations is not important
        primal_lagrangian_store.backward()
        dual_lagrangian_store.backward()

        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.step()
        self.dual_step()

        return RollOut(cmp_state.loss, cmp_state, primal_lagrangian_store, dual_lagrangian_store)
