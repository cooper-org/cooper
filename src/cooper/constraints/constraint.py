from typing import Literal, Optional

from cooper.constraints.constraint_state import ConstraintState
from cooper.formulations import ContributionStore, Formulation, LagrangianFormulation
from cooper.multipliers import Multiplier, PenaltyCoefficient
from cooper.utils import ConstraintType


class Constraint:
    """This class is used to define a constraint in the optimization problem.

    Args:
        constraint_type: One of `cooper.ConstraintType.EQUALITY` or
            `cooper.ConstraintType.INEQUALITY`.
        multiplier: The Lagrange multiplier associated with the constraint.
        formulation_type: The type of formulation for the constrained optimization
            problem. Must be a subclass of :py:class:`~cooper.formulations.Formulation`.
            The default is :py:class:`~cooper.formulations.LagrangianFormulation`.
        penalty_coefficient: The penalty coefficient used to penalize the constraint
            violation. This is only used for some formulations, such as the
            :py:class:`~cooper.formulations.AugmentedLagrangianFormulation`.
    """

    def __init__(
        self,
        constraint_type: ConstraintType,
        multiplier: Multiplier,
        formulation_type: type[Formulation] = LagrangianFormulation,
        penalty_coefficient: Optional[PenaltyCoefficient] = None,
    ) -> None:
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

    def __repr__(self) -> str:
        repr_ = f"constraint_type={self.constraint_type}, formulation={self.formulation}, multiplier={self.multiplier}"
        if self.penalty_coefficient is not None:
            repr_ += f", penalty_coefficient={self.penalty_coefficient}"
        return f"Constraint({repr_})"
