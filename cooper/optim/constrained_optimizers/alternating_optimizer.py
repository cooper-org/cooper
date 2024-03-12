# coding: utf8
"""
Implementation of constrained optimizers based on alternation such as
:py:class:`AlternatingPrimalDualOptimizer`, :py:class:`AlternatingDualPrimalOptimizer`,
:py:class:`AugmentedLagrangianPrimalDualOptimizer` and
:py:class:`AugmentedLagrangianDualPrimalOptimizer`.
"""

import warnings
from typing import Callable, Optional

import torch

from cooper.cmp import CMPState, LagrangianStore
from cooper.formulations import AugmentedLagrangianFormulation
from cooper.multipliers import Multiplier
from cooper.utils import OneOrSequence

from ..types import AlternationType
from .constrained_optimizer import ConstrainedOptimizer


class BaseAlternatingOptimizer(ConstrainedOptimizer):

    extrapolation = False
    alternation_type: AlternationType
    is_augmented_lagrangian_optimizer: bool

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
        Perform sanity checks on the initialization of ``AlternatingOptimizer``.

        Warns:
            UserWarning: The Augmented Lagrangian Method requires all dual optimizers to
                be SGD(lr=1.0).
            UserWarning: Using an AlternatingOptimizer together with multipliers that
                have ``restart_on_feasible=True`` is untested.

        """

        if self.is_augmented_lagrangian_optimizer:
            for dual_optimizer in self.dual_optimizers:
                all_lrs = [_["lr"] for _ in dual_optimizer.param_groups]
                if (dual_optimizer.__class__.__name__ != "SGD") or not all([lr == 1.0 for lr in all_lrs]):
                    warnings.warn("The Augmented Lagrangian Method requires all dual optimizers to be SGD(lr=1.0).")

        for multiplier in self.multipliers:
            if getattr(multiplier, "restart_on_feasible", False):
                warnings.warn("Using alternating updates with dual restarts is untested. Please use with caution.")

    def step(self):
        pass

    def update_penalty_coefficients(self, cmp_state: CMPState) -> None:
        """Update the penalty coefficients of the constraint groups. Only the penalty
        coefficients associated with the ``AugmentedLagrangianFormulation`` and
        constraints that ``contributes_to_dual_update`` are updated.
        """
        for constraint_group, constraint_state in cmp_state.observed_constraints:
            if constraint_group.formulation_type == AugmentedLagrangianFormulation:
                # We might reach this point via an AugmetedLagrangianOptimizer acting
                # on some constraints that do not use an Augmented Lagrangian formulation,
                # so we do _not_ apply penalty coefficient updates to those.
                if constraint_state.contributes_to_dual_update:
                    constraint_group.update_penalty_coefficient(constraint_state=constraint_state)


class AlternatingPrimalDualOptimizer(BaseAlternatingOptimizer):
    """Optimizes a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`
    by performing primal-dual alternating updates to the primal and dual variables.
    """

    # TODO(gallego-posada): Add equations to illustrate the alternating update

    alternation_type = AlternationType.PRIMAL_DUAL
    is_augmented_lagrangian_optimizer = False

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

        self.zero_grad()
        cmp_state = compute_cmp_state_fn()

        # Update primal variables only
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

        lagrangian_store_for_dual = new_cmp_state.populate_dual_lagrangian()
        new_cmp_state.dual_backward()
        self.dual_step(call_extrapolation=False)

        if self.is_augmented_lagrangian_optimizer:
            self.update_penalty_coefficients(cmp_state=new_cmp_state)

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


class AlternatingDualPrimalOptimizer(BaseAlternatingOptimizer):
    """Optimizes a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`
    by performing dual-primal alternating updates to the primal and dual variables.
    """

    # TODO(gallego-posada): Add equations to illustrate the alternating update

    alternation_type = AlternationType.DUAL_PRIMAL
    is_augmented_lagrangian_optimizer = False

    def roll(self, compute_cmp_state_fn: Callable[..., CMPState]) -> tuple[CMPState, LagrangianStore]:
        r"""Performs a dual-primal alternating step where the dual variables are
        updated first (:math:`\lambda_t \\to \lambda_{t+1}`, :math:`\mu_t \\to \mu_{t+1}`),
        and the primal variables are updated (:math:`x_t \\to x_{t+1}`) based on the
        Lagrangian computed with the updated dual variables.

        Note that, unlike for the ``AlternatingPrimalDualOptimizer``, the constraint
        violations only need to be computed once, since the primal variables are not
        changed during the update to the dual variables.

        .. note::
            The Lagrangian value contained in the LagrangianStore at the end of the roll is:

            .. math::
                \mathcal{L} = f(x_t) + {\lambda_{t+1}}^{\\top} g(x_t) + {\mu_{t+1}}^{\\top} h(x_t)

        Args:
            compute_cmp_state_fn: ``Callable`` for evaluating the CMPState.

        """

        self.zero_grad()
        cmp_state = compute_cmp_state_fn()

        # Update dual variables only
        lagrangian_store_for_dual = cmp_state.populate_dual_lagrangian()
        cmp_state.dual_backward()
        self.dual_step(call_extrapolation=False)

        if self.is_augmented_lagrangian_optimizer:
            self.update_penalty_coefficients(cmp_state=cmp_state)

        # Update primal variables based on the Lagrangian at the new dual point, and the
        # objective and constraint violations measured at the old primal point.
        self.zero_grad()
        lagrangian_store_for_primal = cmp_state.populate_primal_lagrangian()
        cmp_state.primal_backward()
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.step()

        # Patch the CMPState and LagrangianStore with the latest available loss and
        # Lagrangian estimate. See the docstring for more details.
        assert lagrangian_store_for_primal.dual_lagrangian is None
        lagrangian_store_for_primal.dual_lagrangian = lagrangian_store_for_dual.dual_lagrangian
        assert lagrangian_store_for_primal.dual_constraint_stores == []
        lagrangian_store_for_primal.dual_constraint_stores = lagrangian_store_for_dual.dual_constraint_stores

        return cmp_state, lagrangian_store_for_primal


class AugmentedLagrangianPrimalDualOptimizer(AlternatingPrimalDualOptimizer):
    """Optimizes a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`
    by performing primal-dual updates according to the Augmented Lagrangian Method.
    """

    # TODO(gallego-posada): Add equations to illustrate the alternating update
    is_augmented_lagrangian_optimizer = True


class AugmentedLagrangianDualPrimalOptimizer(AlternatingDualPrimalOptimizer):
    """Optimizes a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`
    by performing dual-primal updates according to the Augmented Lagrangian Method.
    """

    # TODO(gallego-posada): Add equations to illustrate the alternating update
    is_augmented_lagrangian_optimizer = True
