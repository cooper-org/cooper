from typing import Optional

import torch

from cooper import multipliers
from cooper.constraints.constraint_state import ConstraintState, ConstraintStore, ConstraintType
from cooper.formulations import FormulationType
from cooper.multipliers import Multiplier, PenaltyCoefficient


class ConstraintGroup:
    """Constraint Group."""

    # TODO(gallego-posada): Add documentation
    # TODO(gallego-posada): add docstring explaining that when passing the multiplier
    # directly, the other kwargs (shape, dtype, device) are ignored

    def __init__(
        self,
        constraint_type: ConstraintType,
        formulation_type: Optional[FormulationType] = FormulationType.LAGRANGIAN,
        multiplier: Optional[Multiplier] = None,
        multiplier_kwargs: Optional[dict] = {},
        penalty_coefficient: Optional[PenaltyCoefficient] = None,
    ):
        if not isinstance(constraint_type, ConstraintType):
            raise ValueError(
                f"Expected `constraint_type` to be of type {ConstraintType}, but received {type(constraint_type)}"
            )
        if not isinstance(formulation_type, FormulationType):
            raise ValueError(
                f"Expected `formulation_type` to be of type {FormulationType}, but received {type(formulation_type)}"
            )

        self.constraint_type = constraint_type

        formulation_class = formulation_type.value
        formulation_kwargs = {"constraint_type": self.constraint_type}

        if formulation_class.expects_multiplier:
            if multiplier is None:
                multiplier = multipliers.build_explicit_multiplier(constraint_type, **multiplier_kwargs)
            self.sanity_check_multiplier(multiplier)
            formulation_kwargs["multiplier"] = multiplier
        else:
            if multiplier is not None:
                raise ValueError(f"Formulation {formulation_type} does not admit multipliers.")

        if formulation_class.expects_penalty_coefficient:
            if penalty_coefficient is None:
                raise ValueError(f"Formulation {formulation_type} expects a penalty coefficient but none was provided.")
            formulation_kwargs["penalty_coefficient"] = penalty_coefficient
        else:
            if penalty_coefficient is not None:
                raise ValueError(f"Formulation {formulation_type} does not admit penalty coefficients.")

        self.formulation = formulation_class(**formulation_kwargs)

    def sanity_check_multiplier(self, multiplier: Multiplier) -> None:
        if isinstance(multiplier, multipliers.ExplicitMultiplier):
            if multiplier.implicit_constraint_type != self.constraint_type:
                raise ValueError(f"Provided multiplier is inconsistent with {self.constraint_type} constraint.")

    @property
    def multiplier(self) -> Optional[Multiplier]:
        if hasattr(self.formulation, "multiplier"):
            return self.formulation.multiplier
        else:
            return None

    @property
    def penalty_coefficient(self) -> Optional[PenaltyCoefficient]:
        if hasattr(self.formulation, "penalty_coefficient"):
            return self.formulation.penalty_coefficient
        else:
            return None

    def update_penalty_coefficient(self, value: torch.Tensor) -> None:
        """Update the penalty coefficient of the constraint group."""
        if not hasattr(self.formulation, "penalty_coefficient"):
            raise ValueError(f"Constraint group {self.constraint_type} does not have a penalty coefficient.")
        else:
            self.penalty_coefficient.value = value

    def compute_constraint_contribution(
        self, constraint_state: Optional[ConstraintState] = None
    ) -> tuple[ConstraintStore]:
        """Compute the contribution of the current constraint to the primal and dual
        Lagrangians."""

        if constraint_state is None:
            raise ValueError("A `ConstraintState` is needed to compute Lagrangian contribution")

        return self.formulation.compute_lagrangian_contribution(constraint_state=constraint_state)

    def state_dict(self):
        return {"constraint_type": self.constraint_type, "formulation": self.formulation.state_dict()}

    def load_state_dict(self, state_dict):
        self.constraint_type = state_dict["constraint_type"]
        self.formulation.load_state_dict(state_dict["formulation"])

    def __repr__(self):
        return f"ConstraintGroup(constraint_type={self.constraint_type}, formulation={self.formulation})"
