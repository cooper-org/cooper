"""Lagrangian formulation"""

import abc
import pdb
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, no_type_check

import torch

from .lagrangian_formulation import BaseLagrangianFormulation, LagrangianFormulation
from .multipliers import DenseMultiplier
from .problem import CMPState, ConstrainedMinimizationProblem, Formulation


class AugmentedLagrangianFormulation(LagrangianFormulation):
    """
    Provides utilities for computing the Augmented Lagrangian associated with a
    ``ConstrainedMinimizationProblem`` and for populating the gradients for the
    primal and dual parameters.

    Args:
        cmp: ``ConstrainedMinimizationProblem`` we aim to solve and which gives
            rise to the Lagrangian.
        ineq_init: Initialization values for the inequality multipliers.
        eq_init: Initialization values for the equality multipliers.
        aug_lag_coefficient: Coefficient used for the augmented Lagrangian.

    # TODO: Add mathematical formulation here
    """

    def __init__(
        self,
        cmp: ConstrainedMinimizationProblem,
        ineq_init: Optional[torch.Tensor] = None,
        eq_init: Optional[torch.Tensor] = None,
        aug_lag_coefficient: float = 0.0,
    ):
        """Construct new `AugmentedLagrangianFormulation`"""

        super().__init__(cmp=cmp, ineq_init=ineq_init, eq_init=eq_init)

        assert (
            cmp.is_constrained
        ), "Attempted to create an Augmented Lagrangian formulation for an unconstrained \
            problem. Consider using an `UnconstrainedFormulation` instead."

        if aug_lag_coefficient < 0:
            raise ValueError("Augmented Lagrangian coefficient must be non-negative.")

        self.aug_lag_coefficient = aug_lag_coefficient

    def weighted_violation(
        self, cmp_state: CMPState, constraint_type: str
    ) -> torch.Tensor:
        """
        Computes the dot product between the current multipliers and the
        constraint violations of type ``constraint_type``. If proxy-constraints
        are provided in the :py:class:`.CMPState`, the non-proxy (usually
        non-differentiable) constraints are used for computing the dot product,
        while the "proxy-constraint" dot products are stored under
        ``self.state_update``.

        Args:
            cmp_state: current ``CMPState``
            constraint_type: type of constrained to be used

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
            violation_for_update = torch.sum(multipliers * defect.detach())
            self.state_update.append(violation_for_update)

        return proxy_violation, sq_proxy_violation

    @no_type_check
    def composite_objective(
        self,
        closure: Callable[..., CMPState],
        *closure_args,
        pre_computed_state: Optional[CMPState] = None,
        write_state: bool = True,
        **closure_kwargs
    ) -> torch.Tensor:
        """
        Computes the Lagrangian based on a new evaluation of the
        :py:class:`~cooper.problem.CMPState``.

        If no explicit proxy-constraints are provided, we use the given
        inequality/equality constraints to compute the Lagrangian and to
        populate the primal and dual gradients.

        In case proxy constraints are provided in the CMPState, the non-proxy
        constraints (potentially non-differentiable) are used for computing the
        Lagrangian, while the proxy-constraints are used in the backward
        computation triggered by :py:meth:`._populate_gradient` (and thus must
        be differentiable).

        Args:
            closure: Callable returning a :py:class:`cooper.problem.CMPState`
            pre_computed_state: Pre-computed CMP state to avoid wasteful
                computation when only dual gradients are required.
            write_state: If ``True``, the ``state`` of the formulation's
                :py:class:`cooper.problem.ConstrainedMinimizationProblem`
                attribute is replaced by that returned by the ``closure``
                argument. This flag can be used (when set to ``False``) to
                evaluate the Lagrangian, e.g. for logging validation metrics,
                without overwritting the information stored in the formulation's
                :py:class:`cooper.problem.ConstrainedMinimizationProblem`.

        """

        if pre_computed_state is not None:
            cmp_state = pre_computed_state
        else:
            cmp_state = closure(*closure_args, **closure_kwargs)
        if write_state:
            self.cmp.state = cmp_state

        # Extract values from ProblemState object
        loss = cmp_state.loss

        if self.cmp.is_constrained and (not self.is_state_created):
            # If not done before, instantiate and initialize dual variables
            self.create_state(cmp_state)

        # Compute Lagrangian based on current loss and values of multipliers
        self.purge_state_update()

        if self.cmp.is_constrained:
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

            if self.aug_lag_coefficient > 0:
                # If using augmented Lagrangian, add squared sum of constraints
                # Following the formulation on Marc Toussaint slides (p 17-20)
                # https://ipvs.informatik.uni-stuttgart.de/mlr/marc/teaching/13-Optimization/03-constrainedOpt.pdf
                lagrangian += (
                    0.5 * self.aug_lag_coefficient * (sq_ineq_viol + sq_eq_viol)
                )

        else:
            assert cmp_state.loss is not None
            lagrangian = cmp_state.loss

        return lagrangian
