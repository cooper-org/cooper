"""Lagrangian formulation"""

from typing import Callable, Optional, no_type_check

import torch

from cooper.problem import CMPState, ConstrainedMinimizationProblem

from .lagrangian import LagrangianFormulation


class AugmentedLagrangianFormulation(LagrangianFormulation):
    """
    Provides utilities for computing the Augmented Lagrangian associated with a
    ``ConstrainedMinimizationProblem`` and for populating the gradients for the
    primal and dual parameters accordingly.

    Args:
        cmp: ``ConstrainedMinimizationProblem`` we aim to solve and which gives
            rise to the Lagrangian.
        ineq_init: Initialization values for the inequality multipliers.
        eq_init: Initialization values for the equality multipliers.

    """

    def __init__(
        self,
        cmp: Optional[ConstrainedMinimizationProblem] = None,
        ineq_init: Optional[torch.Tensor] = None,
        eq_init: Optional[torch.Tensor] = None,
    ):
        """Construct new `AugmentedLagrangianFormulation`"""

        super().__init__(cmp=cmp, ineq_init=ineq_init, eq_init=eq_init)

    def weighted_violation(
        self, cmp_state: CMPState, constraint_type: str
    ) -> torch.Tensor:
        """
        Computes the dot product between the current multipliers and the
        constraint violations of type ``constraint_type``. If proxy-constraints
        are provided in the :py:class:`.CMPState`, the non-proxy (usually
        non-differentiable) constraints are used for computing the dot product,
        while the "proxy-constraint" dot products are stored under
        ``self.accumulated_violation_dot_prod``.

        If the ``CMPState`` contains proxy _inequality_ constraints, the
        filtering on whether the constraint is active for the calculation of the
        Augmented Lagrangian is done based on the value of the non-proxy
        constraints.

        Args:
            cmp_state: current ``CMPState``.
            constraint_type: type of constrained to be used, e.g. "eq" or "ineq".
        """

        defect = getattr(cmp_state, constraint_type + "_defect")
        has_defect = defect is not None

        proxy_defect = getattr(cmp_state, "proxy_" + constraint_type + "_defect")
        has_proxy_defect = proxy_defect is not None

        if not has_proxy_defect:
            # If not given proxy constraints, then the regular defects are
            # used for computing gradients and evaluating the multipliers
            proxy_defect = defect

        if not has_defect:
            # We should always have at least the regular defects, if not, then
            # the problem instance does not have `constraint_type` constraints
            proxy_violation = torch.tensor([0.0], device=cmp_state.loss.device)
            sq_proxy_violation = torch.tensor([0.0], device=cmp_state.loss.device)
        else:
            multipliers = getattr(self, constraint_type + "_multipliers")()

            if constraint_type == "ineq":
                # Compute filter based on non-proxy constraint defect
                const_filter = torch.logical_or(defect >= 0, multipliers > 0).detach()
            else:
                # Equality constraints do not need to be filtered
                const_filter = 1.0

            # We compute (primal) gradients of this object
            sq_proxy_violation = torch.sum(const_filter * (proxy_defect) ** 2)

            # We compute (primal) gradients of this object
            proxy_violation = torch.sum(multipliers.detach() * proxy_defect)

            # This is the violation of the "actual" constraint. We use this
            # to update the value of the multipliers by lazily filling the
            # multiplier gradients in `populate_gradients`

            # TODO (JGP): Verify that call to backward is general enough for
            # Lagrange Multiplier models
            violation_for_update = torch.sum(multipliers * defect.detach())
            self.update_accumulated_violation(update=violation_for_update)

        return proxy_violation, sq_proxy_violation

    @no_type_check
    def composite_objective(
        self,
        aug_lag_coeff_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        closure: Callable[..., CMPState] = None,
        *closure_args,
        pre_computed_state: Optional[CMPState] = None,
        write_state: Optional[bool] = True,
        **closure_kwargs
    ) -> torch.Tensor:
        """
        Computes the Lagrangian based on a new evaluation of the
        :py:class:`~cooper.problem.CMPState` via the ``closure`` function.

        If no explicit proxy-constraints are provided, we use the given
        inequality/equality constraints to compute the Augmented Lagrangian and
        to populate the primal and dual gradients. Note that gradients are _not_
        populated by this function, but rather :py:meth:`._populate_gradient`.

        In case proxy constraints are provided in the CMPState, the non-proxy
        constraints (potentially non-differentiable) are used for computing the
        value of the Augmented Lagrangian. The accumulated proxy-constraints
        are used in the backward computation triggered by
        :py:meth:`._populate_gradient` (and thus must be differentiable).

        Args:
            closure: Callable returning a :py:class:`cooper.problem.CMPState`
            pre_computed_state: Pre-computed CMP state to avoid wasteful
                computation when only dual gradients are required.
            write_state: If ``True``, the ``state`` of the formulation's
                :py:class:`cooper.problem.ConstrainedMinimizationProblem`
                attribute is replaced by that returned by the ``closure``
                argument. This flag can be used (when set to ``False``) to
                evaluate the Augmented Lagrangian, e.g. for logging validation
                metrics, without overwritting the information stored in the
                formulation's
                :py:class:`cooper.problem.ConstrainedMinimizationProblem`.

        """

        assert (
            closure is not None or pre_computed_state is not None
        ), "At least one of closure or pre_computed_state must be provided"

        if pre_computed_state is not None:
            cmp_state = pre_computed_state
        else:
            cmp_state = closure(*closure_args, **closure_kwargs)

        if write_state and self.cmp is not None:
            self.write_cmp_state(cmp_state)

        # Extract values from ProblemState object
        loss = cmp_state.loss

        if not self.is_state_created:
            # If not done before, instantiate and initialize dual variables
            self.create_state(cmp_state)

        # Purge previously accumulated constraint violations
        self.update_accumulated_violation(update=None)

        # Compute Augmented Lagrangian based on current loss and values of multipliers

        # Compute contribution of the constraint violations, weighted by the
        # current multiplier values

        # If given proxy constraints, these are used to compute the terms
        # added to the Lagrangian, and the multiplier updates are based on
        # the non-proxy violations.
        # If not given proxy constraints, then gradients and multiplier
        # updates are based on the "regular" constraints.
        ineq_viol, sq_ineq_viol = self.weighted_violation(cmp_state, "ineq")
        eq_viol, sq_eq_viol = self.weighted_violation(cmp_state, "eq")

        # Lagrangian = loss + \sum_i multiplier_i * defect_i
        lagrangian = loss + ineq_viol + eq_viol

        # Gather all the learning rates for the "parameter groups" of the dual
        # variables, and check that all the learning rates are the same.
        dual_lrs = aug_lag_coeff_scheduler.get_last_lr()
        is_all_dual_lr_equal = all(x == dual_lrs[0] for x in dual_lrs)
        assert is_all_dual_lr_equal, "All the dual LRs must be the same."

        # Use the dual learning as the Augmented Lagrangian coefficient to
        # ensure that gradient-based update will coincide with the update
        # scheme of the Augmented Lagrangian method.
        augmented_lagrangian_coefficient = dual_lrs[0]
        if augmented_lagrangian_coefficient > 0:
            # If using augmented Lagrangian, add squared sum of constraints
            # Following the formulation on Marc Toussaint slides (p 17-20)
            # https://ipvs.informatik.uni-stuttgart.de/mlr/marc/teaching/13-Optimization/03-constrainedOpt.pdf
            lagrangian += (
                0.5 * augmented_lagrangian_coefficient * (sq_ineq_viol + sq_eq_viol)
            )

        return lagrangian
