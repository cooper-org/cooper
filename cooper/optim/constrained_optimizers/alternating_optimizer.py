# coding: utf8
"""
Implementation of the :py:class:`AlternatingPrimalDualOptimizer` and
:py:class:`AlternatingPrimalDualOptimizer` classes.
"""

import warnings
from enum import Enum
from typing import Callable, Optional

import torch

from cooper.cmp import CMPState, LagrangianStore
from cooper.multipliers import Multiplier
from cooper.utils import OneOrSequence

from ..types import AlternationType
from .constrained_optimizer import ConstrainedOptimizer


class AlternatingPrimalDualOptimizer(ConstrainedOptimizer):
    """Optimizes a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`
    by performing primal-dual alternating updates to the primal and dual variables.
    """

    # TODO(gallego-posada): Add equations to illustrate the alternating update

    extrapolation = False
    alternation_type = AlternationType.PRIMAL_DUAL

    def __init__(
        self,
        primal_optimizers: OneOrSequence[torch.optim.Optimizer],
        dual_optimizers: OneOrSequence[torch.optim.Optimizer],
        multipliers: Optional[OneOrSequence[Multiplier]] = None,
    ):
        super().__init__(primal_optimizers, dual_optimizers, multipliers)

        self.base_sanity_checks()

        self.custom_sanity_checks()

    def custom_sanity_checks(self):
        """
        Perform sanity checks on the initialization of ``AlternatingPrimalDualOptimizer``.

        Warns:
            UserWarning: Using an AlternatingPrimalDualOptimizer together with
                multipliers that have ``restart_on_feasible=True`` is untested.
        """

        for multiplier in self.multipliers:
            if getattr(multiplier, "restart_on_feasible", False):
                warnings.warn("Using alternating updates with dual restarts is untested. Please use with caution.")

    def step(self):
        pass

    def roll(
        self,
        compute_cmp_state_fn: Callable[..., CMPState],
        compute_violations_fn: Optional[Callable[..., CMPState]] = None,
    ) -> tuple[CMPState, LagrangianStore]:
        r"""Performs a primal-dual alternating step where the primal variables are
        updated first (:math:`x_t \\to x_{t+1}`), and the dual variables are updated
        (:math:`\lambda_t \\to \lambda_{t+1}`, :math:`\mu_t \\to \mu_{t+1}`) based on the
        constraint violations at the updated primal point :math:`x_{t+1}`.

        Note that the constraint violations must be computed twice: once for the initial
        primal update, and once more for the dual update. The second computation can
        exploit the fact that the objective function does not need to be re-evaluated,
        and so the computation can be sped up by only computing the constraint
        violations via the ``compute_violations_fn`` argument.

        .. note::
            Since the ``compute_violations_fn`` argument allows to avoid re-evaluating
            the objective function, we may not have enough information to exactly
            compute the value of the Lagrangian before the dual update
            :math:`\mathcal{L}(x_{t+1}, \lambda_t, \mu_t)`, since only the constraints are
            evaluated at the new primal point :math:`x_{t+1}`.

            Thus we `approximate the Lagrangian` by using the objective function
            computed at the old iterate :math:`x_t`. However, we do have a new estimate
            of the constraint violations :math:`g(x_{t+1})` and :math:`h(x_{t+1})`. The
            Lagrangian value contained in the LagrangianStore at the end of the roll is:

            .. math::
                \mathcal{L} = f(x_t) + {\lambda_{t}}^{\\top} g(x_{t+1}) + {\mu_{t}}^{\\top} h(x_{t+1})

            Whenever the ``compute_violations_fn`` argument is **not** provided, the
            Lagrangian value is computed exactly as:

            .. math::
                \mathcal{L} = f(x_{t+1}) + {\lambda_{t}}^{\\top} g(x_{t+1}) + {\mu_{t}}^{\\top} h(x_{t+1})

            Note that in either case, the returned Lagrangian value uses the value of
            the multipliers before their update, i.e., :math:`\lambda_t` and
            :math:`\mu_t`.

        Args:
            compute_cmp_state_fn: ``Callable`` for evaluating the CMPState.

            compute_violations_fn: ``Callable`` for re-evaluating the constraint
                violations when performing alternating updates. When this argument
                is provided, it takes precedence over the ``compute_cmp_state_fn`` for
                the update of the dual variables. If not provided, the violation
                measured by ``compute_cmp_state_fn`` at the updated primal iterate are
                used. Defaults to None.

        """

        # Update primal variables only
        self.zero_grad()
        cmp_state = compute_cmp_state_fn()

        # TODO(gallego-posada): After the modularization of the populate_lagrangian
        # method of the CMPState, we can make code more efficient by only computing
        # the required primal/dual lagrangians as needed.

        lagrangian_store_for_primal = cmp_state.populate_primal_lagrangian()
        cmp_state.primal_backward()
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.step()

        # Update dual variables based on constraint violations at new primal point
        self.zero_grad()
        with torch.no_grad():
            # Note that the dual variables do not intervene in the computation of the
            # CMP state. This means we can skip gradient computation wrt the primal
            # parameters to avoid wasteful computation, since we will only need the
            # gradient wrt the dual variables.
            if compute_violations_fn is not None:
                new_cmp_state = compute_violations_fn()

                if new_cmp_state.loss is not None:
                    raise RuntimeError(
                        "Expected `compute_violations_fn` to not populate the loss. Please provide this value for the `compute_cmp_state_fn` instead."
                    )
            else:
                new_cmp_state = compute_cmp_state_fn()

            if new_cmp_state._dual_lagrangian is not None:
                error_message = (
                    "Expected a fresh CMPState for alternating update but the provided state has a non-None value"
                    " for the `_dual_lagrangian` attribute."
                )
                raise RuntimeError(error_message)

        # lagrangian_store_post_primal_update = new_cmp_state.populate_lagrangian()
        lagrangian_store_for_dual = new_cmp_state.populate_dual_lagrangian()
        new_cmp_state.dual_backward()
        self.dual_step(call_extrapolation=False)

        # Purge the primal lagrangian to avoid reusing it in the next primal update
        new_cmp_state.purge_primal_lagrangian()

        # Patch the CMPState and LagrangianStore with the latest available loss and
        # Lagrangian estimate. See the docstring for more details.
        if new_cmp_state.loss is None:
            # If the loss was not populated by the `compute_violations_fn`, we copy the
            # loss evaluated during the primal update so users can access it for logging
            new_cmp_state.loss = cmp_state.loss
        assert lagrangian_store_for_dual.lagrangian is None
        lagrangian_store_for_dual.lagrangian = new_cmp_state.loss + lagrangian_store_for_dual.dual_lagrangian
        assert lagrangian_store_for_dual.primal_constraint_stores == []
        lagrangian_store_for_dual.primal_constraint_stores = lagrangian_store_for_primal.primal_constraint_stores

        return new_cmp_state, lagrangian_store_for_dual


class AlternatingDualPrimalOptimizer(ConstrainedOptimizer):
    """Optimizes a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`
    by performing dual-primal alternating updates to the primal and dual variables.
    """

    # TODO(gallego-posada): Add equations to illustrate the alternating update

    extrapolation = False
    alternation_type = AlternationType.DUAL_PRIMAL

    def __init__(
        self,
        primal_optimizers: OneOrSequence[torch.optim.Optimizer],
        dual_optimizers: OneOrSequence[torch.optim.Optimizer],
        multipliers: Optional[OneOrSequence[Multiplier]] = None,
    ):
        super().__init__(primal_optimizers, dual_optimizers, multipliers)

        self.base_sanity_checks()

        self.custom_sanity_checks()

    def custom_sanity_checks(self):
        """
        Perform sanity checks on the initialization of ``AlternatingDualPrimalOptimizer``.

        Warns:
            UserWarning: Using an AlternatingDualPrimalOptimizer together with
                multipliers that have ``restart_on_feasible=True`` is untested.
        """

        for multiplier in self.multipliers:
            if getattr(multiplier, "restart_on_feasible", False):
                warnings.warn("Using alternating updates with dual restarts is untested. Please use with caution.")

    def step(self):
        pass

    def roll(self, compute_cmp_state_fn: Callable[..., CMPState]) -> tuple[CMPState, LagrangianStore]:
        """Performs a dual-primal alternating step where the dual variables are
        updated first, and the primal variables are updated based on the Lagrangian
        computed at the updated dual point. Note that the objective function and
        constraint violations are only computed once, since the primal variables do not
        change during the dual update.

        Args:
            compute_cmp_state_fn: ``Callable`` for evaluating the CMPState.
        """
        self.zero_grad()

        # This cmp_state is shared for both the dual and primal updates
        cmp_state = compute_cmp_state_fn()

        # TODO(gallego-posada): After the modularization of the populate_lagrangian
        # method of the CMPState, we can make code more efficient by only computing
        # the required primal/dual lagrangians as needed.
        # Update dual variables only
        lagrangian_store = cmp_state.populate_lagrangian()
        cmp_state.dual_backward()
        self.dual_step(call_extrapolation=False)

        # Update primal variables based on the Lagrangian at the new dual point, and the
        # objective and constraint violations measured at the old primal point.
        self.zero_grad()
        # Purge primal Lagrangian which was populated during the dual update
        cmp_state.purge_primal_lagrangian()

        # TODO(gallego-posada): After the modularization of the populate_lagrangian
        # method of the CMPState, we can make code more efficient by only computing
        # the required primal/dual lagrangians as needed.
        lagrangian_store_post_dual_step = cmp_state.populate_lagrangian()
        cmp_state.primal_backward()
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.step()
        # Purge the dual lagrangian to avoid reusing it in the next dual update
        cmp_state.purge_dual_lagrangian()

        return cmp_state, lagrangian_store_post_dual_step
