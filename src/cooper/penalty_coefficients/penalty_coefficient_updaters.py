from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import torch

from cooper.penalty_coefficients.penalty_coefficients import DensePenaltyCoefficient, IndexedPenaltyCoefficient
from cooper.utils import ConstraintType

if TYPE_CHECKING:
    from cooper.constraints import Constraint, ConstraintState


class PenaltyCoefficientUpdater(abc.ABC):
    """Abstract class for updating the penalty coefficient of a constraint."""

    def step(self, observed_constraints: dict[Constraint, ConstraintState]) -> None:
        r"""Trigger updates on the penalty coefficients for each of the ``observed_constraints``.

        For each constraint in ``observed_constraints``, this method determines whether
        its penalty coefficient should be updated. The decision depends on properties
        like whether the constraint contributes to primal/dual updates and the
        availability of strict violation measurements.

        Note:
        **Primal vs Dual Contributions**:
        - For formulations expecting multipliers (e.g., AugmentedLagrangian), updates occur if:
            - The constraint contributes to the dual update, **OR**
            - It contributes to the primal update **and** has a strict violation measurement.
        - For primal-only formulations (e.g., QuadraticPenalty), updates occur only if
          the constraint contributes to the primal update.

        TODO(merajhashemi): Add to documentation "We do not update penalty coefficients for constraints that are not observed."
        Args:
            observed_constraints: Dictionary with :py:class:`~Constraint` instances as
                keys and :py:class:`~ConstraintState` instances as values (containing
                tensors :math:`\vg(\vx_t)` and :math:`\vh(\vx_t)`).
        """
        for constraint, constraint_state in observed_constraints.items():
            if constraint.penalty_coefficient is None:
                # Skip constraints without penalty coefficients
                continue

            if constraint.formulation.expects_multiplier:
                # If the user provides a "surrogate" as the `violation` in the constraint state, and does not
                # provide a `strict_violation`, and the constraint is marked as contributing to the dual update, our
                # convention is that said violation is a dual-valid measurement, and thus can be relied on for
                # updating the penalty coefficient.
                contributes_to_dual = constraint_state.contributes_to_dual_update

                # If we only have the `violation` but not the `strict_violation` we cannot be certain that the given
                # `violation` is not a surrogate. Therefore, we cannot rely on it to update the penalty coefficients.
                # On the other hand, if `strict_violation` is given, it can be used to update the penalty coefficients.
                contributes_to_primal = constraint_state.contributes_to_primal_update
                has_strict_violation = constraint_state.strict_violation is not None

                should_update = contributes_to_dual or (contributes_to_primal and has_strict_violation)
            else:
                # If we have a primal only formulation (like QuadraticPenalty),
                # `constraint_state.contributes_to_dual_update` must be `False`.
                # Therefore, we only update the penalty coefficient if
                # `constraint_state.contributes_to_primal_update=True`.
                should_update = constraint_state.contributes_to_primal_update

            if should_update:
                self.update_penalty_coefficient_(constraint, constraint_state)

    @abc.abstractmethod
    def update_penalty_coefficient_(self, constraint: Constraint, constraint_state: ConstraintState) -> None:
        """Update the penalty coefficient of a constraint.

        Args:
            constraint: The constraint for which the penalty coefficient is updated.
            constraint_state: The constraint state of the constraint.
        """


class MultiplicativePenaltyCoefficientUpdater(PenaltyCoefficientUpdater):
    """Multiplicative penalty coefficient updater for Augmented Lagrangian formulation.
    The penalty coefficient is updated by multiplying it by a growth factor when the constraint
    violation is larger than a given tolerance.
    Based on Algorithm 17.4 in Numerical Optimization by Nocedal and Wright.

    Args:
        growth_factor: The factor by which the penalty coefficient is multiplied when the
            constraint is violated beyond ``violation_tolerance``.
        violation_tolerance: The tolerance for the constraint violation. If the violation
            is smaller than this tolerance, the penalty coefficient is not updated.
            The comparison is done at the constraint-level (i.e., each entry of the
            violation tensor). For equality constraints, the absolute violation is
            compared to the tolerance. All constraint types use the strict violation
            (when available) for the comparison.
        has_restart: Whether to restart the penalty coefficient to its initial value when
            the inequality constraint is satisfied. This is only applicable to inequality
            constraints.

    TODO: Add raises docs for ValueErrors
    """

    def __init__(
        self, growth_factor: float = 1.01, violation_tolerance: float = 1e-4, has_restart: bool = True
    ) -> None:
        if violation_tolerance < 0.0:
            raise ValueError("Violation tolerance must be non-negative.")

        self.growth_factor = growth_factor
        self.violation_tolerance = violation_tolerance
        self.has_restart = has_restart

    def update_penalty_coefficient_(self, constraint: Constraint, constraint_state: ConstraintState) -> None:
        """TODO: Add raises docs for TypeErrors"""
        _, strict_violation = constraint_state.extract_violations()
        _, strict_constraint_features = constraint_state.extract_constraint_features()
        penalty_coefficient = constraint.penalty_coefficient

        if isinstance(penalty_coefficient, DensePenaltyCoefficient):
            observed_penalty_values = penalty_coefficient()
        elif isinstance(penalty_coefficient, IndexedPenaltyCoefficient):
            observed_penalty_values = penalty_coefficient(strict_constraint_features)
        else:
            raise TypeError(f"Unsupported penalty coefficient type: {type(penalty_coefficient)}")

        if constraint.constraint_type == ConstraintType.INEQUALITY:
            # For inequality constraints, we only consider the non-negative part of the violation.
            strict_violation = strict_violation.relu()

        is_scalar = observed_penalty_values.dim() == 0
        violation_measure = strict_violation.norm() if is_scalar else strict_violation.abs()

        # Check where the violation exceeds the allowed tolerance
        violation_exceeds_tolerance = violation_measure > self.violation_tolerance

        # Update the penalty coefficient by multiplying it by the growth factor when
        # the constraint violation is larger than the tolerance. When the violation is
        # less than the tolerance, the penalty value is left unchanged.
        new_value = torch.where(
            violation_exceeds_tolerance, observed_penalty_values * self.growth_factor, observed_penalty_values
        )

        # Restart the penalty coefficient to its initial value if inequality constraint is satisfied.
        if self.has_restart and constraint.constraint_type == ConstraintType.INEQUALITY:
            # The strict violation has relu applied to it, so we can check feasibility by comparing to 0.
            is_feasible = torch.eq(violation_measure, 0)
            new_value = torch.where(is_feasible, penalty_coefficient.init, new_value)

        if isinstance(penalty_coefficient, IndexedPenaltyCoefficient) and new_value.dim() > 0:
            penalty_coefficient.value[strict_constraint_features] = new_value.detach()
        else:
            penalty_coefficient.value = new_value.detach()


class AdditivePenaltyCoefficientUpdater(PenaltyCoefficientUpdater):
    """Additive penalty coefficient updater for Augmented Lagrangian formulation.
    The penalty coefficient is updated by adding a constant value when the constraint
    violation is larger than a given tolerance.

    Args:
        increment: The constant value by which the penalty coefficient is added when the
            constraint is violated beyond ``violation_tolerance``.
        violation_tolerance: The tolerance for the constraint violation. If the violation
            is smaller than this tolerance, the penalty coefficient is not updated.
            The comparison is done at the constraint-level (i.e., each entry of the
            violation tensor). For equality constraints, the absolute violation is
            compared to the tolerance. All constraint types use the strict violation
            (when available) for the comparison.
        has_restart: Whether to restart the penalty coefficient to its initial value when
            the inequality constraint is satisfied. This is only applicable to inequality
            constraints.

    TODO: Add raises docs for ValueErrors
    """

    def __init__(self, increment: float = 1.0, violation_tolerance: float = 1e-4, has_restart: bool = True) -> None:
        if violation_tolerance < 0.0:
            raise ValueError("Violation tolerance must be non-negative.")

        self.increment = increment
        self.violation_tolerance = violation_tolerance
        self.has_restart = has_restart

    def update_penalty_coefficient_(self, constraint: Constraint, constraint_state: ConstraintState) -> None:
        """TODO: Add raises docs for TypeErrors"""
        _, strict_violation = constraint_state.extract_violations()
        _, strict_constraint_features = constraint_state.extract_constraint_features()
        penalty_coefficient = constraint.penalty_coefficient

        if isinstance(penalty_coefficient, DensePenaltyCoefficient):
            observed_penalty_values = penalty_coefficient()
        elif isinstance(penalty_coefficient, IndexedPenaltyCoefficient):
            observed_penalty_values = penalty_coefficient(strict_constraint_features)
        else:
            raise TypeError(f"Unsupported penalty coefficient type: {type(penalty_coefficient)}")

        if constraint.constraint_type == ConstraintType.INEQUALITY:
            # For inequality constraints, we only consider the non-negative part of the violation.
            strict_violation = strict_violation.relu()

        is_scalar = observed_penalty_values.dim() == 0
        violation_measure = strict_violation.norm() if is_scalar else strict_violation.abs()

        # Check where the violation exceeds the allowed tolerance
        violation_exceeds_tolerance = violation_measure > self.violation_tolerance

        # Increment penalty coefficients where violations exceed tolerance, leave unchanged otherwise
        new_value = torch.where(
            violation_exceeds_tolerance, observed_penalty_values + self.increment, observed_penalty_values
        )

        # Restart the penalty coefficient to its initial value if inequality constraint is satisfied.
        if self.has_restart and constraint.constraint_type == ConstraintType.INEQUALITY:
            # The strict violation has relu applied to it, so we can check feasibility by comparing to 0.
            is_feasible = torch.eq(violation_measure, 0)
            new_value = torch.where(is_feasible, penalty_coefficient.init, new_value)

        if isinstance(penalty_coefficient, IndexedPenaltyCoefficient) and new_value.dim() > 0:
            penalty_coefficient.value[strict_constraint_features] = new_value.detach()
        else:
            penalty_coefficient.value = new_value.detach()
