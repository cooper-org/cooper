"""Implementation of the :py:class:`ExtrapolationConstrainedOptimizer` class."""

from typing import Optional

import torch

from cooper.optim.constrained_optimizers.constrained_optimizer import ConstrainedOptimizer
from cooper.optim.optimizer import RollOut


class ExtrapolationConstrainedOptimizer(ConstrainedOptimizer):
    r"""Optimizes a :py:class:`~cooper.ConstrainedMinimizationProblem` by performing
    extrapolation updates to the primal and dual variables.

    Given the choice of primal and dual optimizers, an **extrapolation** step is performed
    first:

    .. math::

        \vx_{t+\frac{1}{2}} &= \texttt{primal_optimizer_update} \left( \vx_{t},
            \nabla_{\vx} \Lag_{\text{primal}}(\vx, \vlambda_t, \vmu_t)|_{\vx=\vx_t}
            \right)

        \vlambda_{t+\frac{1}{2}} &= \left[ \texttt{dual_optimizer_update} \left(
            \vlambda_{t}, \nabla_{\vlambda} \Lag_{\text{dual}}({\vx_{t}}, \vlambda,
            \vmu_t) |_{\vlambda=\vlambda_t} \right) \right]_+

        \vmu_{t+\frac{1}{2}} &= \texttt{dual_optimizer_update} \left( \vmu_{t},
            \nabla_{\vmu} \Lag_{\text{dual}}({\vx_{t}}, \vlambda_{t}, \vmu)
            |_{\vmu=\vmu_t} \right).

    This is followed by an **update** step, which modifies the primal and dual variables
    from step :math:`t`, based on the gradients *computed at the extrapolated points*
    :math:`t+\frac{1}{2}`:

    .. math::

        \vx_{t+1} &= \texttt{primal_optimizer_update} \left( \vx_{t}, \nabla_{\vx}
            \Lag_{\text{primal}} \left(\vx, \vlambda_{\color{red} t+\frac{1}{2}},
            \vmu_{\color{red} t+\frac{1}{2}} \right)|_{\vx=\vx_{\color{red}
            t+\frac{1}{2}}} \right)

        \vlambda_{t+1} &= \left[ \texttt{dual_optimizer_update} \left(
            \vlambda_{t}, \nabla_{\vlambda} \Lag_{\text{dual}} \left({\vx_{\color{red}
            t+\frac{1}{2}}}, \vlambda, \vmu_{\color{red} t+\frac{1}{2}} \right)
            |_{\vlambda=\vlambda_{\color{red} t+\frac{1}{2}}}\right) \right]_+

        \vmu_{t+1} &= \texttt{dual_optimizer_update} \left( \vmu_{t}, \nabla_{\vmu}
            \Lag_{\text{dual}}\left({\vx_{\color{red} t+\frac{1}{2}}},
            \vlambda_{\color{red} t+\frac{1}{2}}, \vmu \right) |_{\vmu=\vmu_{\color{red}
            t+\frac{1}{2}}} \right).

    For example, if the primal optimizer is gradient descent and the dual optimizer is
    gradient ascent, the extrapolation step leads to:

    .. math::

        \vx_{t+\frac{1}{2}} &= \vx_t - \eta_{\vx} \left [ \nabla_{\vx} f(\vx_t) +
            \vlambda_t^\top \nabla_{\vx} \vg(\vx_t) + \vmu_t^\top \nabla_{\vx}
            \vh(\vx_t) \right ],

        \vlambda_{t+\frac{1}{2}} &= \left [ \vlambda_t + \eta_{\vlambda}  \vg(\vx_{t})
            \right ]_+,

        \vmu_{t+\frac{1}{2}} &= \vmu_t + \eta_{\vmu} \vh(\vx_t).

    The update step then yields:

    .. math::

        \vx_{t+1} &= \vx_t - \eta_{\vx} \left [ \nabla_{\vx} f \left(\vx_{\color{red}
            t+\frac{1}{2}}\right) + \vlambda_{\color{red} t+\frac{1}{2}}^\top
            \nabla_{\vx} \vg \left(\vx_{\color{red} t+\frac{1}{2}} \right) +
            \vmu_{\color{red} t+\frac{1}{2}}^\top \nabla_{\vx} \vh\left(\vx_{\color{red}
            t+\frac{1}{2}} \right) \right ],

        \vlambda_{t+1} &= \left [ \vlambda_{t+\frac{1}{2}} + \eta_{\vlambda}
            \vg(\vx_{\color{red} t+\frac{1}{2}}) \right ]_+,

        \vmu_{t+1} &= \vmu_{t+\frac{1}{2}} + \eta_{\vmu} \vh(\vx_{\color{red}
            t+\frac{1}{2}}).

    The :py:meth:`~cooper.optim.constrained_optimizers.ExtrapolationConstrainedOptimizer.roll()`
    will simultaneously call the
    :py:meth:`~cooper.optim.torch_optimizers.ExtragradientOptimizer.extrapolation()`
    and :py:meth:`~cooper.optim.torch_optimizers.ExtragradientOptimizer.step()`
    methods of the primal and dual optimizers.

    """

    def custom_sanity_checks(self) -> None:
        """Perform custom sanity checks on the initialization of the optimizer.

        Raises:
            RuntimeError: Tried to construct an
                :py:class:`ExtrapolationConstrainedOptimizer` but some of the provided
                optimizers do not have an extrapolation method.
        """
        are_primal_extra_optims = [hasattr(_, "extrapolation") for _ in self.primal_optimizers]
        are_dual_extra_optims = [hasattr(_, "extrapolation") for _ in self.dual_optimizers]

        if not all(are_primal_extra_optims) or not all(are_dual_extra_optims):
            raise RuntimeError(
                """Some of the provided optimizers do not have an extrapolation method.
                Please ensure that all optimizers are extrapolation capable."""
            )

    @torch.no_grad()
    def primal_extrapolation_step(self) -> None:
        """Perform an extrapolation step on the parameters associated with the primal
        variables.
        """
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.extrapolation()

    @torch.no_grad()
    def dual_extrapolation_step(self) -> None:
        """Perform an extrapolation step on the parameters associated with the dual
        variables.

        After being updated by the dual optimizer steps, the multipliers are
        post-processed (e.g. to ensure non-negativity for inequality constraints).
        """
        # Update multipliers based on current constraint violations (gradients)
        # For unobserved constraints the gradient is None, so this is a no-op.
        for dual_optimizer in self.dual_optimizers:
            dual_optimizer.extrapolation()

        for multiplier in self.cmp.multipliers():
            multiplier.post_step_()

    def roll(self, compute_cmp_state_kwargs: Optional[dict] = None) -> RollOut:
        r"""Performs a full update step on the primal and dual variables.

        Note that the forward and backward computations are carried out
        *twice*, as part of the
        :py:meth:`~cooper.optim.torch_optimizers.ExtragradientOptimizer.extrapolation()`
        and :py:meth:`~cooper.optim.torch_optimizers.ExtragradientOptimizer.step()`
        calls.

        Args:
            compute_cmp_state_kwargs: Keyword arguments to pass to the
                :py:meth:`~cooper.ConstrainedMinimizationProblem.compute_cmp_state()`
                method.

        Returns:
            :py:class:`~cooper.optim.optimizer.RollOut`: A named tuple containing the
            following objects:

            - loss (:py:class:`~torch.Tensor`):
                The loss value computed after the extrapolation step :math:`f(\vx_{t})`.
            - cmp_state (:py:class:`~cooper.CMPState`):
                The CMP state at :math:`\vx_{t}`.
            - primal_lagrangian_store (:py:class:`~cooper.LagrangianStore`):
                The primal Lagrangian store at :math:`\vx_{t}`,
                :math:`\vlambda_{t}` and :math:`\vmu_{t}`.
            - dual_lagrangian_store (:py:class:`~cooper.LagrangianStore`):
                The dual Lagrangian store at :math:`\vx_{t}`, :math:`\vlambda_t` and
                :math:`\vmu_t`.

        .. note::

            The `RollOut` for this scheme returns the loss and `CMPState` values at the
            original point :math:`(\vx_t, \vlambda_t)`, *before* any of the updates are
            performed.
        """
        if compute_cmp_state_kwargs is None:
            compute_cmp_state_kwargs = {}
        for call_extrapolation in (True, False):
            self.zero_grad()
            cmp_state = self.cmp.compute_cmp_state(**compute_cmp_state_kwargs)

            primal_lagrangian_store = cmp_state.compute_primal_lagrangian()
            dual_lagrangian_store = cmp_state.compute_dual_lagrangian()

            if call_extrapolation:
                roll_out = RollOut(cmp_state.loss, cmp_state, primal_lagrangian_store, dual_lagrangian_store)

            primal_lagrangian_store.backward()
            dual_lagrangian_store.backward()

            if call_extrapolation:
                self.primal_extrapolation_step()
                self.dual_extrapolation_step()
            else:
                self.primal_step()
                self.dual_step()

        return roll_out
