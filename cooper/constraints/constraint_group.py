from typing import Optional

import torch

from cooper import multipliers
from cooper.constraints.constraint_state import ConstraintState, ConstraintStore, ConstraintType
from cooper.formulations import Formulation, FormulationType
from cooper.multipliers import Multiplier, PenaltyCoefficient


class ConstraintGroup:
    """Constraint Group."""

    # TODO(gallego-posada): Add documentation

    def __init__(
        self,
        constraint_type: ConstraintType,
        formulation_type: Optional[FormulationType] = FormulationType.LAGRANGIAN,
        multiplier: Optional[Multiplier] = None,
        penalty_coefficient: Optional[PenaltyCoefficient] = None,
    ):

        self.constraint_type = constraint_type
        self.formulation = formulation_type.value(constraint_type=self.constraint_type)

        self.multiplier = multiplier
        if multiplier is not None:
            self.sanity_check_multiplier(multiplier=self.multiplier, constraint_type=self.constraint_type)
            self.sanity_check_multiplier(multiplier=self.multiplier, constraint_type=self.constraint_type)
            if not self.formulation.expects_multiplier:
                ValueError(f"Formulation {self.formulation} does not admit multipliers.")
        else:
            if self.formulation.expects_multiplier:
                ValueError(f"Formulation {self.formulation} expects a multiplier but none was provided.")

        self.penalty_coefficient = penalty_coefficient
        if penalty_coefficient is not None:
            self.sanity_check_penalty_coefficient(penalty_coefficient=self.penalty_coefficient)
            if not self.formulation.expects_penalty_coefficient:
                ValueError(f"Formulation {self.formulation} does not admit penalty coefficients.")
        else:
            if self.formulation.expects_penalty_coefficient:
                ValueError(f"Formulation {self.formulation} expects a penalty coefficient but none was provided.")

    def sanity_check_multiplier(self, multiplier: Multiplier, constraint_type: ConstraintType) -> None:
        if isinstance(multiplier, multipliers.ExplicitMultiplier):
            if multiplier.implicit_constraint_type != constraint_type:
                raise ValueError(
                    f"Constraint type of provided multiplier is {multiplier.implicit_constraint_type} \
                    which is inconsistent with {constraint_type} set for the constraint group."
                )

    def sanity_check_penalty_coefficient(self, penalty_coefficient: PenaltyCoefficient) -> None:
        if torch.any(penalty_coefficient.value < 0):
            raise ValueError("All entries of the penalty coefficient must be non-negative.")

    def update_penalty_coefficient(self, value: torch.Tensor) -> None:
        """Update the penalty coefficient of the constraint group."""
        if self.penalty_coefficient is None:
            raise ValueError(f"Constraint group does not have a penalty coefficient.")
        else:
            self.penalty_coefficient.value = value

    def compute_constraint_contribution(
        self, constraint_state: ConstraintState
    ) -> tuple[ConstraintStore, ConstraintStore]:
        """Compute the contribution of the current constraint to the primal and dual
        Lagrangians."""

        kwargs = {"constraint_state": constraint_state}
        if self.formulation.expects_multiplier:
            kwargs["multiplier"] = self.multiplier
        if self.formulation.expects_penalty_coefficient:
            kwargs["penalty_coefficient"] = self.penalty_coefficient

        return self.formulation.compute_lagrangian_contributions(**kwargs)

    def state_dict(self):
        state_dict = {"constraint_type": self.constraint_type, "formulation": self.formulation.state_dict()}
        for attr_name, attr in [("multiplier", self.multiplier), ("penalty_coefficient", self.penalty_coefficient)]:
            state_dict[attr_name] = attr.state_dict() if attr is not None else None
        return state_dict

    def load_state_dict(self, state_dict):
        self.constraint_type = state_dict["constraint_type"]
        self.formulation.load_state_dict(state_dict["formulation"])

        if state_dict["multiplier"] is not None and self.multiplier is None:
            raise ValueError("Cannot load multiplier state dict since existing multiplier is `None`.")
        elif state_dict["multiplier"] is None and self.multiplier is not None:
            raise ValueError("Multiplier exists but state dict is `None`.")
        elif state_dict["multiplier"] is not None and self.multiplier is not None:
            self.multiplier.load_state_dict(state_dict["multiplier"])

        if state_dict["penalty_coefficient"] is not None and self.penalty_coefficient is None:
            raise ValueError("Cannot load penalty_coefficient state dict since existing penalty_coefficient is `None`.")
        elif state_dict["penalty_coefficient"] is None and self.penalty_coefficient is not None:
            raise ValueError("Penalty coefficient exists but state dict is `None`.")
        elif state_dict["penalty_coefficient"] is not None and self.penalty_coefficient is not None:
            self.penalty_coefficient.load_state_dict(state_dict["penalty_coefficient"])

    def __repr__(self):
        repr = f"ConstraintGroup(constraint_type={self.constraint_type}, formulation={self.formulation}"
        if self.multiplier is not None:
            repr += f", multiplier={self.multiplier}"
        if self.penalty_coefficient is not None:
            repr += f", penalty_coefficient={self.penalty_coefficient}"
        repr += ")"
        return repr
