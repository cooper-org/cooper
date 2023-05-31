import abc
from enum import Enum
from typing import Optional, Tuple

import torch

from cooper.constraints.constraint_state import ConstraintState, ConstraintType
from cooper.multipliers import MULTIPLIER_TYPE, PenaltyCoefficient


def evaluate_multiplier(multiplier: MULTIPLIER_TYPE, constraint_state: ConstraintState) -> Optional[torch.Tensor]:
    """Evaluate the Lagrange multiplier associated with the constraint group.

    Args:
        constraint_state: The current state of the constraint.
    """
    if multiplier is None:
        multiplier_value = None

    else:
        if constraint_state.constraint_features is None:
            multiplier_value = multiplier()
        else:
            multiplier_value = multiplier(constraint_state.constraint_features)

        if len(multiplier_value.shape) == 0:
            multiplier_value = multiplier_value.unsqueeze(0)

    return multiplier_value


def evaluate_penalty(penalty_coefficient: PenaltyCoefficient, constraint_state: ConstraintState) -> torch.Tensor:
    """Evaluate the penalty coefficient associated with the constraint group.

    Args:
        constraint_state: The current state of the constraint.
    """
    if constraint_state.constraint_features is None:
        penalty_coefficient_value = penalty_coefficient()
    else:
        penalty_coefficient_value = penalty_coefficient(constraint_state.constraint_features)

    return penalty_coefficient_value


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
            raise ValueError("The multiplier tensor must be provided if the dual contribution is not skipped.")

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
    def __init__(
        self, constraint_type: ConstraintType, multiplier: MULTIPLIER_TYPE, penalty_coefficient: PenaltyCoefficient
    ):
        # TODO(gallego-posada): Add documentation
        pass

    def compute_lagrangian_contribution(
        self, constraint_state: ConstraintState
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute the contribution of the current constraint to the primal and dual
        Lagrangians, and evaluates the associated Lagrange multiplier."""

        multiplier_value = None
        weighted_violation_for_primal = None
        weighted_violation_for_dual = None

        return multiplier_value, weighted_violation_for_primal, weighted_violation_for_dual

    def state_dict(self):
        pass

    def load_state_dict(self, state_dict: dict):
        pass

    def __repr__(self):
        pass


class PenalizedFormulation(Formulation):
    def __init__(
        self, constraint_type: ConstraintType, multiplier: MULTIPLIER_TYPE, penalty_coefficient: PenaltyCoefficient
    ):
        if constraint_type != ConstraintType.PENALTY:
            raise ValueError("PenalizedFormulation expects `constraint_type=ConstraintType.PENALTY`.")
        if multiplier is not None:
            raise ValueError("PenalizedFormulation does not admit multipliers.")
        if penalty_coefficient is None:
            raise ValueError("PenalizedFormulation requires penalty coefficients.")
        if torch.any(penalty_coefficient() < 0):
            raise ValueError("The penalty coefficients must all be non-negative.")

        self.constraint_type = constraint_type
        self.multiplier = None
        self.penalty_coefficient = penalty_coefficient

    def compute_lagrangian_contribution(
        self, constraint_state: ConstraintState
    ) -> Tuple[None, Optional[torch.Tensor], None]:
        penalty_coefficient_value = evaluate_penalty(self.penalty_coefficient, constraint_state)
        weighted_violation_for_primal = compute_primal_weighted_violation(penalty_coefficient_value, constraint_state)

        # Penalized formulations have no _trainable_ dual variables, so we adopt the
        # convention of setting these variables to None.
        multiplier_value = None
        weighted_violation_for_dual = None

        return multiplier_value, weighted_violation_for_primal, weighted_violation_for_dual

    def state_dict(self):
        return {"constraint_type": self.constraint_type, "penalty_coefficient": self.penalty_coefficient}

    def load_state_dict(self, state_dict):
        if "multiplier" in state_dict:
            Warning("Unexpected key `multiplier` when loading a PenalizedFormulation.")

        self.constraint_type = state_dict["constraint_type"]
        self.penalty_coefficient.load_state_dict(state_dict["penalty_coefficient"])

    def __repr__(self):
        return (
            f"PenalizedFormulation for {self.constraint_type} constraints. With {self.penalty_coefficient.state_dict()}"
        )


class QuadraticPenaltyFormulation(Formulation):
    # TODO(juan43ramirez): document the difference with the PenalizedFormulation
    def __init__(
        self, constraint_type: ConstraintType, multiplier: MULTIPLIER_TYPE, penalty_coefficient: PenaltyCoefficient
    ):
        if constraint_type == ConstraintType.PENALTY:
            raise ValueError("QuadraticPenaltyFormulation requires either an equality or inequality constraint.")
        if multiplier is not None:
            raise ValueError("PenalizedFormulation does not admit multipliers.")
        if penalty_coefficient is None:
            raise ValueError("QuadraticPenaltyFormulation requires penalty coefficients.")
        if torch.any(penalty_coefficient() < 0):
            raise ValueError("The penalty coefficients must all be non-negative.")

        # TODO(juan43ramirez): Add documentation
        self.constraint_type = constraint_type
        self.multiplier = None
        self.penalty_coefficient = penalty_coefficient

    def compute_lagrangian_contribution(
        self, constraint_state: ConstraintState
    ) -> Tuple[None, Optional[torch.Tensor], None]:
        # Compute quadratic penalty term
        penalty_coefficient_value = evaluate_penalty(self.penalty_coefficient, constraint_state)
        weighted_violation_for_primal = compute_quadratic_penalty(
            penalty_coefficient_value=penalty_coefficient_value,
            constraint_state=constraint_state,
            constraint_type=self.constraint_type,
        )

        # Penalized formulations have no _trainable_ dual variables, so we adopt the
        # convention of setting these variables to None.
        multiplier_value = None
        weighted_violation_for_dual = None

        return multiplier_value, weighted_violation_for_primal, weighted_violation_for_dual

    def state_dict(self):
        return {"constraint_type": self.constraint_type, "penalty_coefficient": self.penalty_coefficient.state_dict()}

    def load_state_dict(self, state_dict):
        if "multiplier" in state_dict:
            Warning("Unexpected key `multiplier` when loading a QuadraticPenaltyFormulation.")

        self.constraint_type = state_dict["constraint_type"]
        self.penalty_coefficient.load_state_dict(state_dict["penalty_coefficient"])

    def __repr__(self):
        return f"QuadraticPenaltyFormulation for {self.constraint_type} constraints. With {self.penalty_coefficient}"


class LagrangianFormulation(Formulation):
    def __init__(
        self, constraint_type: ConstraintType, multiplier: MULTIPLIER_TYPE, penalty_coefficient: PenaltyCoefficient
    ):
        if constraint_type == ConstraintType.PENALTY:
            raise ValueError("LagrangianFormulation requires either an equality or inequality constraint.")
        if penalty_coefficient is not None:
            raise ValueError("LagrangianFormulation does not admit penalty coefficients.")
        if multiplier is None:
            raise ValueError("LagrangianFormulation requires a multiplier.")

        self.constraint_type = constraint_type
        self.multiplier = multiplier
        self.penalty_coefficient = None

    def compute_lagrangian_contribution(
        self, constraint_state: ConstraintState
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        multiplier_value = evaluate_multiplier(self.multiplier, constraint_state)

        weighted_violation_for_primal = compute_primal_weighted_violation(multiplier_value, constraint_state)
        weighted_violation_for_dual = compute_dual_weighted_violation(multiplier_value, constraint_state)

        return multiplier_value, weighted_violation_for_primal, weighted_violation_for_dual

    def state_dict(self):
        return {"constraint_type": self.constraint_type, "multiplier": self.multiplier.state_dict()}

    def load_state_dict(self, state_dict):
        if "penalty_coefficient" in state_dict:
            Warning("Unexpected key `penalty_coefficient` when loading a LagrangianFormulation.")

        self.constraint_type = state_dict["constraint_type"]
        self.multiplier.load_state_dict(state_dict["multiplier"])

    def __repr__(self):
        return "LagrangianFormulation for {self.constraint_type} constraints."


class AugmentedLagrangianFormulation(Formulation):
    def __init__(
        self, constraint_type: ConstraintType, multiplier: MULTIPLIER_TYPE, penalty_coefficient: PenaltyCoefficient
    ):
        if constraint_type == ConstraintType.PENALTY:
            raise ValueError("AugmentedLagrangianFormulation requires either an equality or inequality constraint.")
        if penalty_coefficient is None:
            raise ValueError("AugmentedLagrangianFormulation requires penalty coefficients.")
        if torch.any(penalty_coefficient() < 0):
            raise ValueError("The penalty coefficients must all be non-negative.")
        if multiplier is None:
            raise ValueError("AugmentedLagrangianFormulation requires a multiplier.")

        # TODO(juan43ramirez): Add documentation
        self.constraint_type = constraint_type
        self.multiplier = multiplier
        self.penalty_coefficient = penalty_coefficient

    def compute_lagrangian_contribution(
        self, constraint_state: ConstraintState
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        multiplier_value = evaluate_multiplier(self.multiplier, constraint_state)
        penalty_coefficient_value = evaluate_penalty(self.penalty_coefficient, constraint_state)

        weighted_violation_for_primal = compute_primal_weighted_violation(multiplier_value, constraint_state)
        if weighted_violation_for_primal is not None and not torch.all(penalty_coefficient_value == 0):
            # Compute quadratic penalty term
            penalty = compute_quadratic_penalty(
                penalty_coefficient_value=self.penalty_coefficient(),
                constraint_state=constraint_state,
                constraint_type=self.constraint_type,
            )
            weighted_violation_for_primal += penalty

        # TODO: document. Point is to automatically multiply the learning rate of the
        # penalty coefficient by the penalty coefficient.
        multiplier_value_for_dual = multiplier_value * self.penalty_coefficient()
        weighted_violation_for_dual = compute_dual_weighted_violation(multiplier_value_for_dual, constraint_state)

        return multiplier_value, weighted_violation_for_primal, weighted_violation_for_dual

    def state_dict(self):
        return {
            "constraint_type": self.constraint_type,
            "multiplier": self.multiplier.state_dict(),
            "penalty_coefficient": self.penalty_coefficient.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.constraint_type = state_dict["constraint_type"]
        self.multiplier.load_state_dict(state_dict["multiplier"])
        self.penalty_coefficient.load_state_dict(state_dict["penalty_coefficient"])

    def __repr__(self):
        return f"AugmentedLagrangianFormulation for {self.constraint_type} constraints. Penalty coefficient: {self.penalty_coefficient}"


class FormulationType(Enum):
    PENALTY = PenalizedFormulation
    QUADRATIC_PENALTY = QuadraticPenaltyFormulation
    LAGRANGIAN = LagrangianFormulation
    AUGMENTED_LAGRANGIAN = AugmentedLagrangianFormulation
