# coding: utf8
"""
Implementation of constrained optimizers based on alternation such as
:py:class:`AlternatingPrimalDualOptimizer`, :py:class:`AlternatingDualPrimalOptimizer`,
:py:class:`AugmentedLagrangianPrimalDualOptimizer` and
:py:class:`AugmentedLagrangianDualPrimalOptimizer`.
"""

import warnings

import torch

from cooper.cmp import CMPState, LagrangianStore
from cooper.optim.constrained_optimizers.constrained_optimizer import ConstrainedOptimizer
from cooper.optim.types import AlternationType


class BaseAlternatingOptimizer(ConstrainedOptimizer):

    extrapolation = False
    alternation_type: AlternationType
    is_augmented_lagrangian_optimizer: bool

    def custom_sanity_checks(self):
        """
        Perform sanity checks on the initialization of ``AlternatingOptimizer``.

        Warns:
            # TODO(gallego-posada): Is this warning still accurate?
            UserWarning: The Augmented Lagrangian Method requires all dual optimizers to
                be SGD(lr=1.0).
        """

        if self.is_augmented_lagrangian_optimizer:
            for dual_optimizer in self.dual_optimizers:
                all_lrs = [_["lr"] for _ in dual_optimizer.param_groups]
                if (dual_optimizer.__class__.__name__ != "SGD") or not all([lr == 1.0 for lr in all_lrs]):
                    warnings.warn("The Augmented Lagrangian Method requires all dual optimizers to be SGD(lr=1.0).")


class AlternatingPrimalDualOptimizer(BaseAlternatingOptimizer):
    """Optimizes a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`
    by performing primal-dual alternating updates to the primal and dual variables.
    """

    # TODO(gallego-posada): Add equations to illustrate the alternating update

    alternation_type = AlternationType.PRIMAL_DUAL
    is_augmented_lagrangian_optimizer = False

    def roll(
        self,
        compute_cmp_state_kwargs: dict = {},
        compute_violations_kwargs: dict = {},
    ) -> tuple[CMPState, LagrangianStore, LagrangianStore]:
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
            compute_cmp_state_kwargs: Keyword arguments to pass to the ``compute_cmp_state``
                method.

            compute_violations_kwargs: Keyword arguments to pass to the ``compute_violations``
                method. When the ``compute_violations`` method is implemented, it takes
                precedence over the ``compute_cmp_state`` for the update of the dual
                variables. If not implemented, the violation measured by ``compute_cmp_state``
                at the updated primal iterate are used.

        """

        self.zero_grad()
        cmp_state = self.cmp.compute_cmp_state(**compute_cmp_state_kwargs)

        # Update primal variables only
        primal_lagrangian_store = cmp_state.compute_primal_lagrangian()
        primal_lagrangian_store.backward()
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.step()

        # Update dual variables based on constraint violations at new primal point
        self.zero_grad()
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

        # TODO(gallego-posada): Document that users should inspect primal_lagrangian_store for logging purposes
        return new_cmp_state, primal_lagrangian_store, dual_lagrangian_store


class AlternatingDualPrimalOptimizer(BaseAlternatingOptimizer):
    """Optimizes a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`
    by performing dual-primal alternating updates to the primal and dual variables.
    """

    # TODO(gallego-posada): Add equations to illustrate the alternating update

    alternation_type = AlternationType.DUAL_PRIMAL
    is_augmented_lagrangian_optimizer = False

    def roll(self, compute_cmp_state_kwargs: dict = {}) -> tuple[CMPState, LagrangianStore, LagrangianStore]:
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
            compute_cmp_state_kwargs: Keyword arguments to pass to the ``compute_cmp_state`` method.

        """

        self.zero_grad()
        cmp_state = self.cmp.compute_cmp_state(**compute_cmp_state_kwargs)

        # Update dual variables only
        dual_lagrangian_store = cmp_state.compute_dual_lagrangian()
        dual_lagrangian_store.backward()
        self.dual_step()

        # Update primal variables based on the Lagrangian at the new dual point, and the
        # objective and constraint violations measured at the old primal point.
        self.zero_grad()
        primal_lagrangian_store = cmp_state.compute_primal_lagrangian()
        primal_lagrangian_store.backward()
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.step()

        return cmp_state, primal_lagrangian_store, dual_lagrangian_store


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
