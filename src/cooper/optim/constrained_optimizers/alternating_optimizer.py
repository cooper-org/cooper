"""Implementation of constrained optimizers based on alternation such as
:py:class:`AlternatingPrimalDualOptimizer` and :py:class:`AlternatingDualPrimalOptimizer`.
"""

from typing import Optional

import torch

from cooper.optim.constrained_optimizers.constrained_optimizer import ConstrainedOptimizer
from cooper.optim.optimizer import RollOut


class AlternatingPrimalDualOptimizer(ConstrainedOptimizer):
    r"""Optimizes a :py:class:`~cooper.ConstrainedMinimizationProblem`
    by performing alternating updates, starting with the primal variables.

    According to the choice of primal and dual optimizers, updates are performed as follows:

    .. math::
        \vx_{t+1} &= \texttt{primal_optimizer_update} \left( \vx_{t}, \nabla_{\vx}
            \Lag_{\text{primal}}(\vx, \vlambda_t, \vmu_t)|_{\vx=\vx_t} \right)

        \vlambda_{t+1} &= \left[ \texttt{dual_optimizer_update} \left( \vlambda_{t},
            \nabla_{\vlambda} \Lag_{\text{dual}}({\vx_{\color{red} t+1}}, \vlambda, \vmu_t)|_{\vlambda=\vlambda_t}
            \right) \right]_+

        \vmu_{t+1} &= \texttt{dual_optimizer_update} \left( \vmu_{t}, \nabla_{\vmu}
            \Lag_{\text{dual}}({\vx_{\color{red} t+1}}, \vlambda_t, \vmu)|_{\vmu=\vmu_t} \right)

    For instance, when employing alternating projected gradient descent-ascent on a
    :py:class:`~cooper.formulations.Lagrangian` formulation, the updates are as follows:

    .. math::
        \vx_{t+1} &= \vx_t - \eta_{\vx} \left [ \nabla_{\vx} f(\vx_t) + \vlambda_t^\top
            \nabla_{\vx} \vg(\vx_t) + \vmu_t^\top \nabla_{\vx} \vh(\vx_t) \right ],

        \vlambda_{t+1} &= \left [ \vlambda_t + \eta_{\vlambda}  \vg(\vx_{\color{red} t+1})  \right ]_+,

        \vmu_{t+1} &= \vmu_t + \eta_{\vmu} \vh(\vx_{\color{red} t+1}),

    where :math:`\eta_{\vx}`, :math:`\eta_{\vlambda}`, and :math:`\eta_{\vmu}` are step
    sizes.

    This optimizer computes constraint violations *twice*: at :math:`\vx_{t}` for the
    initial primal update, and again at the updated primal point :math:`\vx_{t+1}`
    to update the dual variables. The former are used to compute the primal
    Lagrangian :math:`\Lag_{\text{primal}}` while the latter are used to compute the
    dual Lagrangian :math:`\Lag_{\text{dual}}`.

    .. admonition:: Reducing computational overhead in primal-dual alternating updates
        :class: note

        To update the dual variables, only the constraint violations
        :math:`\vg(\vx_{\color{red} t+1})` and :math:`\vh(\vx_{\color{red} t+1})` are
        required, not the objective function value :math:`f(\vx_{\color{red} t+1})`. To
        reduce computational overhead, the user can implement the
        :py:meth:`~cooper.ConstrainedMinimizationProblem.compute_violations()`
        method of the CMP and pass the ``compute_violations_kwargs`` argument to
        :py:meth:`roll()`. This approach ensures that only the constraint violations
        are recomputed at :math:`\vx_{\color{red} t+1}`, without calculating the loss or
        constructing a computational graph over the primal variables.
    """

    def roll(
        self, compute_cmp_state_kwargs: Optional[dict] = None, compute_violations_kwargs: Optional[dict] = None
    ) -> RollOut:
        r"""Performs a primal-dual alternating step where the primal variables are
        updated first.

        Args:
            compute_cmp_state_kwargs: Keyword arguments to pass to the
                :py:meth:`~cooper.ConstrainedMinimizationProblem.compute_cmp_state()`
                method.

            compute_violations_kwargs: Keyword arguments to pass to the
                :py:meth:`~cooper.ConstrainedMinimizationProblem.compute_violations()`
                method. When
                :py:meth:`~cooper.ConstrainedMinimizationProblem.compute_violations()`
                is implemented, it takes precedence over
                :py:meth:`~cooper.ConstrainedMinimizationProblem.compute_cmp_state()`
                for the dual update. If not implemented, the violations measured by
                :py:meth:`~cooper.ConstrainedMinimizationProblem.compute_cmp_state()`
                at the updated primal iterate are used.

        Returns:
            :py:class:`~cooper.optim.optimizer.RollOut`: A named tuple containing the
            following objects:

            - loss (:py:class:`~torch.Tensor`):
                The most recent loss value at the end of the roll. If
                :py:meth:`~cooper.ConstrainedMinimizationProblem.compute_violations()`
                was used, returns :math:`f(\vx_{t})`. Otherwise, returns the recomputed
                loss at the updated primal point :math:`f(\vx_{t+1})`.
            - cmp_state (:py:class:`~cooper.CMPState`):
                The CMP state at :math:`\vx_{\color{red} t+1}`. Note that if
                :py:meth:`~cooper.ConstrainedMinimizationProblem.compute_violations()`
                is used, the loss at :math:`\vx_{t+1}` is not computed and
                ``cmp_state.loss`` will be ``None``.
            - primal_lagrangian_store (:py:class:`~cooper.LagrangianStore`):
                The primal Lagrangian store at :math:`\vx_{t}`,
                :math:`\vlambda_t` and :math:`\vmu_t`.
            - dual_lagrangian_store (:py:class:`~cooper.LagrangianStore`):
                The dual Lagrangian store at :math:`\vx_{\color{red} t+1}`,
                :math:`\vlambda_t` and :math:`\vmu_t`.

        """
        if compute_violations_kwargs is None:
            compute_violations_kwargs = {}
        if compute_cmp_state_kwargs is None:
            compute_cmp_state_kwargs = {}
        self.zero_grad()
        cmp_state = self.cmp.compute_cmp_state(**compute_cmp_state_kwargs)

        # Update primal variables only
        primal_lagrangian_store = cmp_state.compute_primal_lagrangian()
        primal_lagrangian_store.backward()
        self.primal_step()

        # Update dual variables based on constraint violations at new primal point
        with torch.no_grad():
            # Note that the dual variables do not intervene in the computation of the
            # CMP state. This means we can skip gradient computation wrt the primal
            # parameters to avoid wasteful computation, since we will only need the
            # gradient wrt the dual variables.
            try:
                new_cmp_state = self.cmp.compute_violations(**compute_violations_kwargs)

                if new_cmp_state.loss is not None:
                    raise RuntimeError(
                        "Expected `compute_violations` to not populate the loss. "
                        "Please provide this value for the `compute_cmp_state` instead."
                    )

            except NotImplementedError:
                new_cmp_state = self.cmp.compute_cmp_state(**compute_cmp_state_kwargs)

        dual_lagrangian_store = new_cmp_state.compute_dual_lagrangian()
        dual_lagrangian_store.backward()
        self.dual_step()

        loss = new_cmp_state.loss if new_cmp_state.loss is not None else cmp_state.loss

        return RollOut(loss, new_cmp_state, primal_lagrangian_store, dual_lagrangian_store)


class AlternatingDualPrimalOptimizer(ConstrainedOptimizer):
    r"""Optimizes a :py:class:`~cooper.ConstrainedMinimizationProblem`
    by performing alternating updates, starting with the dual variables.

    According to the choice of primal and dual optimizers, updates are performed as
    follows:

    .. math::

        \vlambda_{t+1} &= \left[ \texttt{dual_optimizer_update} \left( \vlambda_{t},
            \nabla_{\vlambda} \Lag_{\text{dual}}(\vx_t, \vlambda, \vmu_t)|_{\vlambda=\vlambda_t}
            \right) \right]_+

        \vmu_{t+1} &= \texttt{dual_optimizer_update} \left( \vmu_{t}, \nabla_{\vmu}
            \Lag_{\text{dual}}(\vx_t, \vlambda_t, \vmu)|_{\vmu=\vmu_t} \right)

        \vx_{t+1} &= \texttt{primal_optimizer_update} \left( \vx_{t}, \nabla_{\vx}
            \Lag_{\text{primal}}(\vx, \vlambda_{\color{red} t+1}, \vmu_{\color{red} t+1}
            )|_{\vx=\vx_t} \right)

    For instance, when employing alternating projected gradient descent-ascent on a
    :py:class:`~cooper.formulations.Lagrangian` formulation, the updates are as follows:

    .. math::

        \vlambda_{t+1} &= \left [ \vlambda_t + \eta_{\vlambda}  \vg(\vx_t)  \right ]_+,

        \vmu_{t+1} &= \vmu_t + \eta_{\vmu} \vh(\vx_t),

        \vx_{t+1} &= \vx_t - \eta_{\vx} \left [ \nabla_{\vx} f(\vx_t) +
            \vlambda_{\color{red} t+1}^\top \nabla_{\vx} \vg(\vx_t) +
            \vmu_{\color{red} t+1}^\top \nabla_{\vx} \vh(\vx_t) \right ],

    where :math:`\eta_{\vx}`, :math:`\eta_{\vlambda}`, and :math:`\eta_{\vmu}` are step
    sizes.

    .. note::
        Both the primal and dual updates depend on the :py:class:`~cooper.CMPState` at
        the current primal iterate :math:`\vx_{t}`. Consequently, although the primal
        update uses the updated dual variables :math:`\vlambda_{\color{red} t+1}` and
        :math:`\vmu_{\color{red} t+1}`, the :py:class:`~cooper.CMPState` **does not need
        to be recomputed after the dual update**. As a result, the computational cost of
        this optimizer matches that of the
        :py:class:`~cooper.optim.constrained_optimizers.SimultaneousOptimizer`.
    """

    def roll(self, compute_cmp_state_kwargs: Optional[dict] = None) -> RollOut:
        r"""Performs a dual-primal alternating step where the dual variables are
        updated first.

        Args:
            compute_cmp_state_kwargs: Keyword arguments to pass to the
                :py:meth:`~cooper.ConstrainedMinimizationProblem.compute_cmp_state()`
                method

        Returns:
            :py:class:`~cooper.optim.optimizer.RollOut`: A named tuple containing the
            following objects:

            - loss (:py:class:`~torch.Tensor`):
                The loss value computed during the roll, :math:`f(\vx_{t})`.
            - cmp_state (:py:class:`~cooper.CMPState`):
                The CMP state at :math:`\vx_{t}`.
            - primal_lagrangian_store (:py:class:`~cooper.LagrangianStore`):
                The primal Lagrangian store at :math:`\vx_{t}`,
                :math:`\vlambda_{\color{red} t+1}` and :math:`\vmu_{\color{red} t+1}`.
            - dual_lagrangian_store (:py:class:`~cooper.LagrangianStore`):
                The dual Lagrangian store at :math:`\vx_{t}`, :math:`\vlambda_t` and
                :math:`\vmu_t`.
        """
        if compute_cmp_state_kwargs is None:
            compute_cmp_state_kwargs = {}
        self.zero_grad()
        cmp_state = self.cmp.compute_cmp_state(**compute_cmp_state_kwargs)

        # Update dual variables only
        dual_lagrangian_store = cmp_state.compute_dual_lagrangian()
        dual_lagrangian_store.backward()
        self.dual_step()

        # Update primal variables based on the Lagrangian at the new dual point, and the
        # objective and constraint violations measured at the old primal point.
        primal_lagrangian_store = cmp_state.compute_primal_lagrangian()
        primal_lagrangian_store.backward()
        self.primal_step()

        return RollOut(cmp_state.loss, cmp_state, primal_lagrangian_store, dual_lagrangian_store)
