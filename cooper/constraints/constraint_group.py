from typing import Optional

import torch

from cooper import multipliers
from cooper.constraints.constraint_state import ConstraintState, ConstraintStore, ConstraintType
from cooper.formulations import Formulation, FormulationType
from cooper.multipliers import ExplicitMultiplier, IndexedMultiplier, Multiplier, PenaltyCoefficient


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
        self.formulation_type = formulation_type
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
            if multiplier.constraint_type != constraint_type:
                raise ValueError(
                    f"Constraint type of provided multiplier is {multiplier.constraint_type} \
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

    def prepare_kwargs_for_lagrangian_contribution(self, constraint_state: ConstraintState) -> dict:
        kwargs = {"constraint_state": constraint_state}
        if self.formulation.expects_multiplier:
            kwargs["multiplier"] = self.multiplier
        if self.formulation.expects_penalty_coefficient:
            kwargs["penalty_coefficient"] = self.penalty_coefficient

        return kwargs

    def compute_constraint_primal_contribution(self, constraint_state: ConstraintState) -> ConstraintStore:
        """Compute the contribution of the current constraint to the primal Lagrangian."""
        kwargs = self.prepare_kwargs_for_lagrangian_contribution(constraint_state=constraint_state)
        return self.formulation.compute_contribution_for_primal_lagrangian(**kwargs)

    def compute_constraint_dual_contribution(self, constraint_state: ConstraintState) -> ConstraintStore:
        """Compute the contribution of the current constraint to the dual Lagrangian."""
        kwargs = self.prepare_kwargs_for_lagrangian_contribution(constraint_state=constraint_state)
        return self.formulation.compute_contribution_for_dual_lagrangian(**kwargs)

    def update_strictly_feasible_indices_(
        self, strict_violation: torch.Tensor, constraint_state: ConstraintState
    ) -> None:

        # Determine which of the constraints are strictly feasible and update the
        # `strictly_feasible_indices` attribute of the multiplier.
        if getattr(self.multiplier, "restart_on_feasible", False):

            if isinstance(self.multiplier, IndexedMultiplier):
                # Need to expand the indices to the size of the multiplier
                strictly_feasible_indices = torch.zeros_like(self.multiplier.weight, dtype=torch.bool)

                # IndexedMultipliers have a shape of (-, 1). We need to unsqueeze
                # dimension 1 of the violations
                strictly_feasible_indices[constraint_state.strict_constraint_features] = (
                    strict_violation.unsqueeze(1) < 0.0
                )
            else:
                strictly_feasible_indices = strict_violation < 0.0

            self.multiplier.strictly_feasible_indices = strictly_feasible_indices

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
