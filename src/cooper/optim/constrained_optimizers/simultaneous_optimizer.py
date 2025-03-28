"""Implementation of the :py:class:`SimultaneousOptimizer` class."""

from typing import Optional

from cooper.optim.constrained_optimizers.constrained_optimizer import ConstrainedOptimizer
from cooper.optim.optimizer import RollOut


class SimultaneousOptimizer(ConstrainedOptimizer):
    r"""Optimizes a :py:class:`~cooper.ConstrainedMinimizationProblem`
    by performing simultaneous gradient updates to the primal and dual variables.

    According to the choice of primal and dual optimizers, the updates are performed as follows:

    .. math::
        \vx_{t+1} &= \texttt{primal_optimizer_update} \left( \vx_{t}, \nabla_{\vx}
            \Lag_{\text{primal}}(\vx, \vlambda_t, \vmu_t)|_{\vx=\vx_t} \right)

        \vlambda_{t+1} &= \left[ \texttt{dual_optimizer_update} \left( \vlambda_{t},
            \nabla_{\vlambda} \Lag_{\text{dual}}({\vx_{t}}, \vlambda, \vmu_t)|_{\vlambda=\vlambda_t}
            \right) \right]_+

        \vmu_{t+1} &= \texttt{dual_optimizer_update} \left( \vmu_{t}, \nabla_{\vmu}
            \Lag_{\text{dual}}({\vx_{t}}, \vlambda_t, \vmu)|_{\vmu=\vmu_t} \right)

    For instance, when the primal/dual updates are gradient descent/ascent on a
    :py:class:`~cooper.formulations.Lagrangian` formulation, the updates are as follows:

    .. math::
        \vx_{t+1} &= \vx_t - \eta_{\vx} \left [ \nabla_{\vx} f(\vx_t) + \vlambda_t^\top
            \nabla_{\vx} \vg(\vx_t) + \vmu_t^\top \nabla_{\vx} \vh(\vx_t) \right ],

        \vlambda_{t+1} &= \left [ \vlambda_t + \eta_{\vlambda}  \vg(\vx_t)  \right ]_+,

        \vmu_{t+1} &= \vmu_t + \eta_{\vmu} \vh(\vx_t),

    where :math:`\eta_{\vx}`, :math:`\eta_{\vlambda}`, and :math:`\eta_{\vmu}` are step
    sizes.

    """

    def roll(self, compute_cmp_state_kwargs: Optional[dict] = None) -> RollOut:
        """Evaluates the :py:class:`~cooper.CMPState` and performs a simultaneous
        primal-dual update.

        Args:
            compute_cmp_state_kwargs: Keyword arguments to pass to the
                :py:meth:`~cooper.ConstrainedMinimizationProblem.compute_cmp_state`
                method.
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
