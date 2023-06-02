import abc
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Tuple, Union

import torch

import cooper.formulations.utils as formulation_utils
from cooper.constraints.constraint_state import ConstraintState, ConstraintType
from cooper.multipliers import Multiplier, PenaltyCoefficient, evaluate_constraint_factor


@dataclass
class ConstraintContribution:
    """Stores the value of the factor (multiplier or penalty coefficient), the
    contribution of the constraint to the primal-differentiable Lagrian, and the
    contribution of the constraint to the dual-differentiable Lagrangian."""

    multiplier_value: Optional[torch.Tensor] = None
    penalty_coefficient_value: Optional[torch.Tensor] = None
    primal_contribution: Optional[torch.Tensor] = None
    dual_contribution: Optional[torch.Tensor] = None


class Formulation(abc.ABC):
    # TODO(gallego-posada): Add documentation

    @abc.abstractmethod
    def compute_lagrangian_contribution(self, constraint_state: ConstraintState) -> ConstraintContribution:
        """Computes the contribution from the current constraint to the primal and dual
        Lagrangians, and evaluates the associated Lagrange multiplier or penalty
        coefficient."""
        pass

    @abc.abstractmethod
    def state_dict(self):
        pass

    @abc.abstractmethod
    def load_state_dict(self, state_dict: dict):
        pass


class PenaltyFormulation(Formulation):
    def __init__(self, constraint_type: ConstraintType, penalty_coefficient: PenaltyCoefficient):
        if constraint_type != ConstraintType.PENALTY:
            raise ValueError("PenaltyFormulation expects `constraint_type=ConstraintType.PENALTY`.")
        if penalty_coefficient is None:
            raise ValueError("PenaltyFormulation expects penalty coefficients.")
        if torch.any(penalty_coefficient.value < 0):
            raise ValueError("All penalty coefficients must be non-negative.")

        self.constraint_type = constraint_type
        self.penalty_coefficient = penalty_coefficient

    def compute_lagrangian_contribution(self, constraint_state: ConstraintState) -> ConstraintContribution:
        penalty_coefficient_value = evaluate_constraint_factor(self.penalty_coefficient, constraint_state)

        weighted_violation_for_primal = formulation_utils.compute_primal_weighted_violation(
            constraint_factor=penalty_coefficient_value, constraint_state=constraint_state
        )

        return ConstraintContribution(
            penalty_coefficient_value=penalty_coefficient_value,
            primal_contribution=weighted_violation_for_primal,
            dual_contribution=None,
        )

    def state_dict(self):
        return {"constraint_type": self.constraint_type, "penalty_coefficient": self.penalty_coefficient.state_dict()}

    def load_state_dict(self, state_dict):
        self.constraint_type = state_dict["constraint_type"]
        self.penalty_coefficient.load_state_dict(state_dict["penalty_coefficient"])

    def __repr__(self):
        return f"PenaltyFormulation({self.constraint_type}, penalty_coefficient={self.penalty_coefficient})"


class QuadraticPenaltyFormulation(Formulation):
    # TODO(juan43ramirez): emphasize the difference with respect to the PenaltyFormulation

    def __init__(self, constraint_type: ConstraintType, penalty_coefficient: PenaltyCoefficient):
        # TODO(juan43ramirez): Add documentation

        if constraint_type not in [ConstraintType.EQUALITY, ConstraintType.INEQUALITY]:
            raise ValueError("QuadraticPenaltyFormulation requires an equality or inequality constraint.")
        if penalty_coefficient is None:
            raise ValueError("QuadraticPenaltyFormulation requires a penalty coefficient.")
        if torch.any(penalty_coefficient.value < 0):
            raise ValueError("The penalty coefficients must all be non-negative.")

        self.constraint_type = constraint_type
        self.penalty_coefficient = penalty_coefficient

    def compute_lagrangian_contribution(self, constraint_state: ConstraintState) -> ConstraintContribution:
        penalty_coefficient_value = evaluate_constraint_factor(self.penalty_coefficient, constraint_state)

        weighted_violation_for_primal = formulation_utils.compute_quadratic_penalty(
            penalty_coefficient_value=penalty_coefficient_value,
            constraint_state=constraint_state,
            constraint_type=self.constraint_type,
        )

        return ConstraintContribution(
            penalty_coefficient_value=penalty_coefficient_value,
            primal_contribution=weighted_violation_for_primal,
            dual_contribution=None,
        )

    def state_dict(self):
        return {"constraint_type": self.constraint_type, "penalty_coefficient": self.penalty_coefficient.state_dict()}

    def load_state_dict(self, state_dict):
        self.constraint_type = state_dict["constraint_type"]
        self.penalty_coefficient.load_state_dict(state_dict["penalty_coefficient"])

    def __repr__(self):
        return f"QuadraticPenaltyFormulation({self.constraint_type}, penalty_coefficient={self.penalty_coefficient})"


class LagrangianFormulation(Formulation):
    def __init__(self, constraint_type: ConstraintType, multiplier: Multiplier):
        if constraint_type not in [ConstraintType.EQUALITY, ConstraintType.INEQUALITY]:
            raise ValueError("LagrangianFormulation requires an equality or inequality constraint.")
        if multiplier is None:
            raise ValueError("LagrangianFormulation requires a multiplier.")

        self.constraint_type = constraint_type
        self.multiplier = multiplier

    def compute_lagrangian_contribution(self, constraint_state: ConstraintState) -> ConstraintContribution:
        multiplier_value = evaluate_constraint_factor(self.multiplier, constraint_state)

        weighted_violation_for_primal = formulation_utils.compute_primal_weighted_violation(
            multiplier_value, constraint_state
        )
        weighted_violation_for_dual = formulation_utils.compute_dual_weighted_violation(
            multiplier_value, constraint_state
        )

        return ConstraintContribution(
            multiplier_value=multiplier_value,
            primal_contribution=weighted_violation_for_primal,
            dual_contribution=weighted_violation_for_dual,
        )

    def state_dict(self):
        return {"constraint_type": self.constraint_type, "multiplier": self.multiplier.state_dict()}

    def load_state_dict(self, state_dict):
        self.constraint_type = state_dict["constraint_type"]
        self.multiplier.load_state_dict(state_dict["multiplier"])

    def __repr__(self):
        return f"LagrangianFormulation({self.constraint_type})."


class AugmentedLagrangianFormulation(Formulation):
    def __init__(
        self, constraint_type: ConstraintType, multiplier: Multiplier, penalty_coefficient: PenaltyCoefficient
    ):
        # TODO(juan43ramirez): Add documentation

        if constraint_type == ConstraintType.PENALTY:
            raise ValueError("AugmentedLagrangianFormulation requires either an equality or inequality constraint.")
        if penalty_coefficient is None:
            raise ValueError("AugmentedLagrangianFormulation requires a penalty coefficient.")
        if torch.any(penalty_coefficient.value < 0):
            raise ValueError("The penalty coefficients must all be non-negative.")
        if multiplier is None:
            raise ValueError("AugmentedLagrangianFormulation requires a multiplier.")

        self.constraint_type = constraint_type
        self.multiplier = multiplier
        self.penalty_coefficient = penalty_coefficient

    def compute_lagrangian_contribution(self, constraint_state: ConstraintState) -> ConstraintContribution:

        multiplier_value = evaluate_constraint_factor(self.multiplier, constraint_state)
        penalty_coefficient_value = evaluate_constraint_factor(self.penalty_coefficient, constraint_state)

        weighted_violation_for_primal = formulation_utils.compute_primal_weighted_violation(
            multiplier_value, constraint_state
        )
        if weighted_violation_for_primal is not None and not torch.all(penalty_coefficient_value == 0):
            # FIXME(juan43ramirez): There is an *evaluation* of the penalty coefficient
            # earlier in this function. Previously the code below was making a call to
            # self.penalty_coefficient(). Why was the code not using the evaluated
            # penalty coefficient? Is this a bug? A potential discrepancy would be a
            # problem in the case of IndexedPenaltyCoefficients.
            # The offending line below was:
            #   penalty_coefficient_value=self.penalty_coefficient(),
            weighted_violation_for_primal += formulation_utils.compute_quadratic_penalty(
                penalty_coefficient_value=penalty_coefficient_value,
                constraint_state=constraint_state,
                constraint_type=self.constraint_type,
            )

        # TODO: document. Point is to automatically multiply the learning rate of the
        # penalty coefficient by the penalty coefficient.

        # FIXME(juan43ramirez): Why does the definition of `penalty_coefficient_value`
        # make a call to the evaluate_constraint_factor function, but here the code was
        # using self.penalty_coefficient()? Is this a bug?
        # The offending line below was:
        #   multiplier_value_for_dual = multiplier_value * self.penalty_coefficient()
        multiplier_value_for_dual = multiplier_value * penalty_coefficient_value
        weighted_violation_for_dual = formulation_utils.compute_dual_weighted_violation(
            multiplier_value_for_dual, constraint_state
        )

        return ConstraintContribution(
            multiplier_value=multiplier_value,
            penalty_coefficient_value=penalty_coefficient_value,
            primal_contribution=weighted_violation_for_primal,
            dual_contribution=weighted_violation_for_dual,
        )

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
        return f"AugmentedLagrangianFormulation({self.constraint_type}, penalty_coefficient={self.penalty_coefficient})"
