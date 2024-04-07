import abc

import torch

from cooper.constraints import Constraint, ConstraintState, ConstraintType
from cooper.multipliers import DensePenaltyCoefficient


class PenaltyCoefficientUpdater(abc.ABC):
    """Abstract class for updating the penalty coefficient of a constraint."""

    def step(self, observed_constraints: dict[Constraint, ConstraintState]):
        for constraint, constraint_state in observed_constraints.items():
            # If a constraint does not contribute to the dual update, we do not update
            # its penalty coefficient.
            if constraint_state.contributes_to_dual_update:
                self.update_penalty_coefficient_(constraint, constraint_state)

    @abc.abstractmethod
    def update_penalty_coefficient_(self, constraint: Constraint, constraint_state: ConstraintState) -> None:
        pass


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
    """

    def __init__(self, growth_factor: float = 1.01, violation_tolerance: float = 1e-4):
        if violation_tolerance < 0.0:
            raise ValueError("Violation tolerance must be non-negative.")

        self.growth_factor = growth_factor
        self.violation_tolerance = violation_tolerance

    def update_penalty_coefficient_(self, constraint: Constraint, constraint_state: ConstraintState) -> None:

        violation, strict_violation = constraint_state.extract_violations()
        constraint_features, strict_constraint_features = constraint_state.extract_constraint_features()
        penalty_coefficient = constraint.penalty_coefficient

        values_for_observed = (
            penalty_coefficient.value
            if isinstance(penalty_coefficient, DensePenaltyCoefficient)
            else penalty_coefficient.value[strict_constraint_features]
        )

        if constraint.constraint_type == ConstraintType.EQUALITY:
            condition = strict_violation.abs() > self.violation_tolerance
        else:
            condition = strict_violation > self.violation_tolerance

        new_value = torch.where(condition, values_for_observed * self.growth_factor, values_for_observed)

        if isinstance(penalty_coefficient, DensePenaltyCoefficient):
            penalty_coefficient.value = new_value.detach()
        else:
            penalty_coefficient.value[strict_constraint_features] = new_value.detach()
