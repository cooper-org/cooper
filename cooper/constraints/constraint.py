from typing import Optional

import torch

from cooper.constraints.constraint_state import ConstraintMeasurement, ConstraintState, ConstraintType
from cooper.formulations import Formulation, LagrangianFormulation
from cooper.multipliers import ExplicitMultiplier, IndexedMultiplier, Multiplier, PenaltyCoefficient


class Constraint:
    """Constraint."""

    # TODO(gallego-posada): Add documentation

    def __init__(
        self,
        constraint_type: ConstraintType,
        multiplier: Optional[Multiplier] = None,
        formulation_type: Optional[Formulation] = LagrangianFormulation,
        penalty_coefficient: Optional[PenaltyCoefficient] = None,
    ):

        self.constraint_type = constraint_type
        self.formulation_type = formulation_type
        self.formulation = formulation_type(constraint_type=self.constraint_type)

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
        if isinstance(multiplier, ExplicitMultiplier):
            if multiplier.constraint_type != constraint_type:
                raise ValueError(
                    f"Constraint type of provided multiplier is {multiplier.constraint_type} \
                    which is inconsistent with {constraint_type} set for the constraint."
                )

    def sanity_check_penalty_coefficient(self, penalty_coefficient: PenaltyCoefficient) -> None:
        if torch.any(penalty_coefficient.value < 0):
            raise ValueError("All entries of the penalty coefficient must be non-negative.")

    def update_penalty_coefficient(self, constraint_state: ConstraintState) -> None:
        """Update the penalty coefficient of the constraint."""
        if self.penalty_coefficient is None:
            raise ValueError("Constraint does not have a penalty coefficient.")
        else:
            self.penalty_coefficient.update_value(
                constraint_state=constraint_state,
                constraint_type=self.constraint_type,
                growth_factor=self.formulation.penalty_growth_factor,
                violation_tolerance=self.formulation.violation_tolerance,
            )

    def prepare_kwargs_for_lagrangian_contribution(self, constraint_state: ConstraintState) -> dict:
        kwargs = {"constraint_state": constraint_state}
        if self.formulation.expects_multiplier:
            kwargs["multiplier"] = self.multiplier
        if self.formulation.expects_penalty_coefficient:
            kwargs["penalty_coefficient"] = self.penalty_coefficient

        return kwargs

    def compute_constraint_primal_contribution(
        self, constraint_state: ConstraintState
    ) -> tuple[Optional[torch.Tensor], Optional[ConstraintMeasurement]]:
        """Compute the contribution of the current constraint to the primal Lagrangian."""
        kwargs = self.prepare_kwargs_for_lagrangian_contribution(constraint_state=constraint_state)
        return self.formulation.compute_contribution_for_primal_lagrangian(**kwargs)

    def compute_constraint_dual_contribution(
        self, constraint_state: ConstraintState
    ) -> tuple[Optional[torch.Tensor], Optional[ConstraintMeasurement]]:
        """Compute the contribution of the current constraint to the dual Lagrangian."""
        kwargs = self.prepare_kwargs_for_lagrangian_contribution(constraint_state=constraint_state)
        return self.formulation.compute_contribution_for_dual_lagrangian(**kwargs)

    def update_strictly_feasible_indices_(
        self, strict_violation: torch.Tensor, strict_constraint_features: torch.Tensor
    ) -> None:

        # Determine which of the constraints are strictly feasible and update the
        # `strictly_feasible_indices` attribute of the multiplier.
        if getattr(self.multiplier, "restart_on_feasible", False):

            if isinstance(self.multiplier, IndexedMultiplier):
                # Need to expand the indices to the size of the multiplier
                strictly_feasible_indices = torch.zeros_like(self.multiplier.weight, dtype=torch.bool)

                # IndexedMultipliers have a shape of (-, 1). We need to unsqueeze
                # dimension 1 of the violations
                strictly_feasible_indices[strict_constraint_features] = strict_violation.unsqueeze(1) < 0.0
            else:
                strictly_feasible_indices = strict_violation < 0.0

            self.multiplier.strictly_feasible_indices = strictly_feasible_indices

    def __repr__(self):
        repr = f"Constraint(constraint_type={self.constraint_type}, formulation={self.formulation}"
        if self.multiplier is not None:
            repr += f", multiplier={self.multiplier}"
        if self.penalty_coefficient is not None:
            repr += f", penalty_coefficient={self.penalty_coefficient}"
        repr += ")"
        return repr
