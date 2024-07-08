from typing import Literal, Optional, Type

from cooper.constraints.constraint_state import ConstraintState
from cooper.constraints.constraint_type import ConstraintType
from cooper.formulations import ContributionStore, Formulation, LagrangianFormulation
from cooper.multipliers import Multiplier, PenaltyCoefficient


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
        self.multiplier.set_constraint_type(constraint_type)

        self.penalty_coefficient = penalty_coefficient
        self.formulation.sanity_check_penalty_coefficient(penalty_coefficient)

    def compute_contribution_to_lagrangian(
        self, constraint_state: ConstraintState, primal_or_dual: Literal["primal", "dual"]
    ) -> Optional[ContributionStore]:
        """Compute the contribution of the current constraint to the primal or dual Lagrangian."""
        compute_contribution_fn = getattr(self.formulation, f"compute_contribution_to_{primal_or_dual}_lagrangian")
        return compute_contribution_fn(
            constraint_state=constraint_state,
            multiplier=self.multiplier,
            penalty_coefficient=self.penalty_coefficient,
        )

    def __repr__(self):
        repr = f"constraint_type={self.constraint_type}, formulation={self.formulation}, multiplier={self.multiplier}"
        if self.penalty_coefficient is not None:
            repr += f", penalty_coefficient={self.penalty_coefficient}"
        return f"Constraint({repr})"
