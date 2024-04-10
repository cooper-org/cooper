from typing import Literal, Optional, Type

import torch

from cooper.constraints.constraint_state import ConstraintMeasurement, ConstraintState
from cooper.formulations import Formulation, LagrangianFormulation
from cooper.multipliers import Multiplier, PenaltyCoefficient

from .constraint_type import ConstraintType


class Constraint:
    """Constraint."""

    # TODO(gallego-posada): Add documentation

    def __init__(
        self,
        constraint_type: ConstraintType,
        multiplier: Multiplier,
        formulation_type: Type[Formulation] = LagrangianFormulation,
        penalty_coefficient: Optional[PenaltyCoefficient] = None,
    ):

        self.constraint_type = constraint_type
        self.formulation_type = formulation_type
        self.formulation = formulation_type(constraint_type=self.constraint_type)

        self.multiplier = multiplier
        if multiplier.constraint_type != self.constraint_type:
            raise ValueError(
                f"Attempted to pair {self.constraint_type} constraint, with {multiplier.constraint_type} multiplier."
            )
        self.multiplier.sanity_check()  # TODO: This fails for ImplicitMultiplier

        self.penalty_coefficient = penalty_coefficient
        self.sanity_check_penalty_coefficient(penalty_coefficient=self.penalty_coefficient)

    def sanity_check_penalty_coefficient(self, penalty_coefficient: PenaltyCoefficient) -> None:
        if self.formulation.expects_penalty_coefficient:
            if self.penalty_coefficient is None:
                raise ValueError(f"{self.formulation_type} expects a penalty coefficient but none was provided.")
            else:
                if torch.any(penalty_coefficient.value < 0):
                    raise ValueError("All entries of the penalty coefficient must be non-negative.")
        else:
            if penalty_coefficient is not None:
                raise ValueError(f"Received unexpected penalty coefficient for {self.formulation_type}.")

    def prepare_kwargs_for_lagrangian_contribution(self, constraint_state: ConstraintState) -> dict:
        kwargs = {"constraint_state": constraint_state, "multiplier": self.multiplier}
        if self.formulation.expects_penalty_coefficient:
            kwargs["penalty_coefficient"] = self.penalty_coefficient
        return kwargs

    def compute_contribution_to_lagrangian(
        self, constraint_state: ConstraintState, primal_or_dual: Literal["primal", "dual"]
    ) -> tuple[Optional[torch.Tensor], Optional[ConstraintMeasurement]]:
        """Compute the contribution of the current constraint to the primal or dual Lagrangian."""
        kwargs = self.prepare_kwargs_for_lagrangian_contribution(constraint_state=constraint_state)
        compute_contribution_fn = getattr(self.formulation, f"compute_contribution_to_{primal_or_dual}_lagrangian")
        return compute_contribution_fn(**kwargs)

    def __repr__(self):
        repr = f"constraint_type={self.constraint_type}, formulation={self.formulation}, multiplier={self.multiplier}"
        if self.penalty_coefficient is not None:
            repr += f", penalty_coefficient={self.penalty_coefficient}"
        return f"Constraint({repr})"
