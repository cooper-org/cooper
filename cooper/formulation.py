import abc
from enum import Enum
from typing import Optional, Tuple

import torch

from cooper.constraints.constraint_state import ConstraintState, ConstraintType
from cooper.multipliers import PenaltyCoefficient


def extract_and_patch_violations(constraint_state: ConstraintState):
    """Extracts the violation and strict violation from the constraint state, and
    patches the strict violation if it is not available. We also unsqueeze the
    violation tensors to ensure thay have at least 1-dimension."""

    violation = constraint_state.violation
    if len(violation.shape) == 0:
        violation = violation.unsqueeze(0)

    strict_violation = constraint_state.strict_violation
    if strict_violation is None:
        strict_violation = constraint_state.violation
    if len(strict_violation.shape) == 0:
        strict_violation = strict_violation.unsqueeze(0)

    return violation, strict_violation


def compute_primal_weighted_violation(multiplier_value: torch.Tensor, constraint_state) -> Optional[torch.Tensor]:
    """Computes the sum of constraint violations weighted by the associated multipliers,
    while preserving the gradient for the primal variables.

    Args:
        multiplier_value: The value of the multiplier for the constraint group.
        constraint_state: The current state of the constraint.
    """

    if constraint_state.skip_primal_contribution:
        # Ignore the primal contribution if the constraint is marked as non-contributing
        # to the primal Lagrangian.
        return None
    else:
        violation, _ = extract_and_patch_violations(constraint_state)

        if multiplier_value is None:
            raise ValueError("The multiplier tensor must be provided if the primal contribution is not skipped.")
        if violation is None:
            raise ValueError("The violation tensor must be provided if the primal contribution is not skipped.")

        # When computing the gradient of the Lagrangian with respect to the primal
        # variables, we do not need to differentiate the multiplier.
        return torch.einsum("i...,i...->", multiplier_value.detach(), violation)


def compute_dual_weighted_violation(multiplier_value: torch.Tensor, constraint_state) -> Optional[torch.Tensor]:
    """Computes the sum of constraint violations weighted by the associated multipliers,
    while preserving the gradient for the dual variables.

    When computing the gradient of the Lagrangian with respect to the dual variables, we
    only need the _value_ of the constraint violation and not its gradient. So we detach
    the violation to avoid computing its gradient. Note that this enables the use of
    non-differentiable constraints for updating the multiplier.

    This insight was originally proposed by Cotter et al. in the paper "Optimization
    with Non-Differentiable Constraints with Applications to Fairness, Recall, Churn,
    and Other Goals" under the name of "proxy" constraints.
    (https://jmlr.org/papers/v20/18-616.html, Sec. 4.2)

    Args:
        multiplier_value: The value of the multiplier for the constraint group.
        constraint_state: The current state of the constraint.
    """

    if constraint_state.skip_dual_contribution:
        # Ignore the primal contribution if the constraint is marked as non-contributing
        # to the dual Lagrangian.
        return None
    else:
        if multiplier_value is None:
            raise ValueError("The multiplier tensor must be provided if the primal contribution is not skipped.")

        # Strict violation represents the "actual" violation of the constraint. When
        # provided, we use the strict violation to update the value of the multiplier.
        # Otherwise, we default to using the differentiable violation.
        _, strict_violation = extract_and_patch_violations(constraint_state)

        return torch.einsum("i...,i...->", multiplier_value, strict_violation.detach())


def compute_quadratic_penalty(
    penalty_coefficient_value: torch.Tensor, constraint_state: ConstraintState, constraint_type: ConstraintType
):
    # TODO(juan43ramirez): Add documentation

    if constraint_state.skip_primal_contribution:
        return None
    else:
        violation, strict_violation = extract_and_patch_violations(constraint_state)

        if violation is None:
            raise ValueError("The violation tensor must be provided if the primal contribution is not skipped.")

        if constraint_type == ConstraintType.INEQUALITY:
            # Compute filter based on strict constraint violation
            const_filter = strict_violation >= 0

            # TODO(juan43ramirez): We used to also penalize inequality constraints
            # with a non-zero multiplier. This seems confusing to me.
            # const_filter = torch.logical_or(strict_violation >= 0, multiplier_value.detach() > 0)

            sq_violation = const_filter * (violation**2)

        elif constraint_type == ConstraintType.EQUALITY:
            # Equality constraints do not need to be filtered
            sq_violation = violation**2
        else:
            raise ValueError(f"{constraint_type} is incompatible with quadratic penalties.")

        if penalty_coefficient_value.numel() == 1:
            # One penalty coefficient shared across all constraints.
            return torch.sum(sq_violation) * penalty_coefficient_value / 2
        else:
            # One penalty coefficient per constraint.
            if violation.shape != penalty_coefficient_value.shape:
                raise ValueError("The violation tensor must have the same shape as the penalty coefficient tensor.")

            return torch.einsum("i...,i...->", penalty_coefficient_value, sq_violation) / 2


class Formulation(abc.ABC):
    def __init__(self, *args, **kwargs):
        # TODO(gallego-posada): Add documentation
        pass

    def compute_lagrangian_contribution(self, *args, **kwargs) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        weighted_violation_for_primal = None
        weighted_violation_for_dual = None

        return weighted_violation_for_primal, weighted_violation_for_dual

    def state_dict(self):
        pass

    def load_state_dict(self, state_dict: dict):
        pass

    def __repr__(self):
        pass


class PenalizedFormulation(Formulation):
    def __init__(self, constraint_type: ConstraintType, penalty_coefficient: PenaltyCoefficient):
        if constraint_type == ConstraintType.EQUALITY:
            raise ValueError("PenalizedFormulation expects inequality constraints.")

        if torch.any(penalty_coefficient() < 0):
            raise ValueError("The penalty coefficients must all be non-negative.")

        self.penalty_coefficient = penalty_coefficient
        self.constraint_type = constraint_type

    def compute_lagrangian_contribution(
        self, constraint_state: ConstraintState, **kwargs
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        weighted_violation_for_primal = compute_primal_weighted_violation(self.penalty_coefficient(), constraint_state)
        weighted_violation_for_dual = None

        return weighted_violation_for_primal, weighted_violation_for_dual


class QuadraticPenaltyFormulation(Formulation):
    def __init__(self, constraint_type: ConstraintType, penalty_coefficient: PenaltyCoefficient):
        if torch.any(penalty_coefficient() < 0):
            raise ValueError("The penalty coefficients must all be non-negative.")

        # TODO(juan43ramirez): Add documentation
        self.penalty_coefficient = penalty_coefficient
        self.constraint_type = constraint_type

    def compute_lagrangian_contribution(
        self, constraint_state: ConstraintState, **kwargs
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Compute quadratic penalty term
        weighted_violation_for_primal = compute_quadratic_penalty(
            penalty_coefficient_value=self.penalty_coefficient(),
            constraint_state=constraint_state,
            constraint_type=self.constraint_type,
        )

        # Penalized formulations have no _trainable_ dual variables, so we adopt the
        # convention of setting this variable to None.
        weighted_violation_for_dual = None

        return weighted_violation_for_primal, weighted_violation_for_dual

    def update_state_(self, *args, **kwargs):
        # TODO(juan43ramirez): Could implement a helper function for increasing the
        # penalty coefficient.
        pass

    def state_dict(self):
        return {"penalty_coefficient": self.penalty_coefficient, "constraint_type": self.constraint_type}

    def load_state_dict(self, state_dict):
        self.penalty_coefficient = state_dict["penalty_coefficient"]
        self.constraint_type = state_dict["constraint_type"]

    def __repr__(self):
        return f"QuadraticPenaltyFormulation for {self.constraint_type} constraints. Penalty coefficient: {self.penalty_coefficient}"


class LagrangianFormulation(Formulation):
    def __init__(self, constraint_type: ConstraintType):
        # TODO(juan43ramirez): Add documentation
        self.constraint_type = constraint_type

    def compute_lagrangian_contribution(
        self, constraint_state: ConstraintState, multiplier_value: torch.Tensor, **kwargs
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        weighted_violation_for_primal = compute_primal_weighted_violation(multiplier_value, constraint_state)
        weighted_violation_for_dual = compute_dual_weighted_violation(multiplier_value, constraint_state)

        return weighted_violation_for_primal, weighted_violation_for_dual

    def state_dict(self):
        return {"constraint_type": self.constraint_type}

    def load_state_dict(self, state_dict):
        self.constraint_type = state_dict["constraint_type"]

    def __repr__(self):
        return "LagrangianFormulation for {self.constraint_type} constraints."


class AugmentedLagrangianFormulation(Formulation):
    def __init__(self, constraint_type: ConstraintType, penalty_coefficient: PenaltyCoefficient):
        if torch.any(penalty_coefficient() < 0):
            raise ValueError("The penalty coefficients must all be non-negative.")

        # TODO(juan43ramirez): Add documentation
        self.penalty_coefficient = penalty_coefficient
        self.constraint_type = constraint_type

    def compute_lagrangian_contribution(
        self, constraint_state: ConstraintState, multiplier_value: torch.Tensor, **kwargs
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        weighted_violation_for_primal = compute_primal_weighted_violation(multiplier_value, constraint_state)
        if weighted_violation_for_primal is not None:
            penalty = compute_quadratic_penalty(
                penalty_coefficient_value=self.penalty_coefficient(),
                constraint_state=constraint_state,
                constraint_type=self.constraint_type,
            )
            weighted_violation_for_primal += penalty

        # TODO: document
        multiplier_value_for_dual = multiplier_value * self.penalty_coefficient()
        weighted_violation_for_dual = compute_dual_weighted_violation(multiplier_value_for_dual, constraint_state)

        return weighted_violation_for_primal, weighted_violation_for_dual

    def state_dict(self):
        return {"penalty_coefficient": self.penalty_coefficient, "constraint_type": self.constraint_type}

    def load_state_dict(self, state_dict):
        self.penalty_coefficient = state_dict["penalty_coefficient"]
        self.constraint_type = state_dict["constraint_type"]

    def __repr__(self):
        return f"AugmentedLagrangianFormulation for {self.constraint_type} constraints. Penalty coefficient: {self.penalty_coefficient}"


class FormulationType(Enum):
    PENALTY = PenalizedFormulation
    QUADRATIC_PENALTY = QuadraticPenaltyFormulation
    LAGRANGIAN = LagrangianFormulation
    AUGMENTED_LAGRANGIAN = AugmentedLagrangianFormulation
