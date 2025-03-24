from typing import Literal, Optional

from cooper.constraints.constraint_state import ConstraintState
from cooper.formulations import ContributionStore, Formulation, Lagrangian
from cooper.multipliers import Multiplier
from cooper.penalty_coefficients import PenaltyCoefficient
from cooper.utils import ConstraintType


class Constraint:
    """A constraint in a constrained optimization problem.

    Args:
        constraint_type: One of :py:class:`cooper.ConstraintType.EQUALITY` or
            :py:class:`cooper.ConstraintType.INEQUALITY`.
        formulation_type: The formulation type for computing the constraint's contribution
            to the Lagrangian. Defaults to :py:class:`~cooper.formulations.Lagrangian`.
        multiplier: The Lagrange multiplier associated with the constraint. This is only
            used for formulations with ``Formulation.expects_multiplier=True``, such as
            the :py:class:`~cooper.formulations.Lagrangian`.
        penalty_coefficient: The penalty coefficient used to penalize the constraint
            violation. This is only used for formulations with
            ``Formulation.expects_penalty_coefficient=True``, such as the
            :py:class:`~cooper.formulations.AugmentedLagrangian`.
    """

    def __init__(
        self,
        constraint_type: ConstraintType,
        formulation_type: type[Formulation] = Lagrangian,
        multiplier: Optional[Multiplier] = None,
        penalty_coefficient: Optional[PenaltyCoefficient] = None,
    ) -> None:
        self._name = None

        self.constraint_type = constraint_type
        self.formulation_type = formulation_type
        self.formulation = formulation_type(constraint_type=self.constraint_type)

        self.multiplier = multiplier
        self.formulation.sanity_check_multiplier(multiplier)
        if self.multiplier is not None:
            self.multiplier.set_constraint_type(constraint_type)

        self.penalty_coefficient = penalty_coefficient
        self.formulation.sanity_check_penalty_coefficient(penalty_coefficient)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        if self._name is not None:
            raise ValueError("Cannot set the name of a constraint more than once.")
        self._name = name

    def compute_contribution_to_lagrangian(
        self, constraint_state: ConstraintState, primal_or_dual: Literal["primal", "dual"]
    ) -> Optional[ContributionStore]:
        """Compute the contribution of the current constraint to the primal or dual Lagrangian."""
        compute_contribution_fn = getattr(self.formulation, f"compute_contribution_to_{primal_or_dual}_lagrangian")

        kwargs = {"constraint_state": constraint_state}
        if self.formulation.expects_penalty_coefficient:
            kwargs["penalty_coefficient"] = self.penalty_coefficient
        if self.formulation.expects_multiplier:
            kwargs["multiplier"] = self.multiplier
        return compute_contribution_fn(**kwargs)

    def __repr__(self) -> str:
        repr_ = f"constraint_type={self.constraint_type}, formulation={self.formulation}"
        if self.multiplier is not None:
            repr_ += f", multiplier={self.multiplier}"
        if self.penalty_coefficient is not None:
            repr_ += f", penalty_coefficient={self.penalty_coefficient}"
        return f"Constraint({repr_})"
