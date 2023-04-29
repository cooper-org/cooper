from typing import Iterator, Optional, Sequence, Tuple, Union

import torch

from cooper import multipliers
from cooper.constraints.constraint_state import ConstraintState, ConstraintType
from cooper.formulation import FormulationType
from cooper.multipliers import MULTIPLIER_TYPE


class ConstraintGroup:
    """Constraint Group."""

    # TODO(gallego-posada): Add documentation
    # TODO(gallego-posada): add docstring explaining that when passing the multiplier
    # directly, the other kwargs (shape, dtype, device) are ignored

    def __init__(
        self,
        constraint_type: ConstraintType,
        formulation_type: Optional[FormulationType] = FormulationType.LAGRANGIAN,
        formulation_kwargs: Optional[dict] = {},
        multiplier: Optional[MULTIPLIER_TYPE] = None,
        multiplier_kwargs: Optional[dict] = {},
    ):

        self._state: ConstraintState = None

        self.constraint_type = constraint_type
        self.formulation = self.build_formulation(formulation_type, formulation_kwargs)

        if multiplier is None:
            multiplier = multipliers.build_explicit_multiplier(constraint_type, **multiplier_kwargs)
        self.sanity_check_multiplier(multiplier)
        self.multiplier = multiplier

    def build_formulation(self, formulation_type, formulation_kwargs):
        if self.constraint_type == ConstraintType.PENALTY and formulation_type != FormulationType.PENALIZED:
            raise ValueError("Constraint of type `penalty` requires a `penalized` formulation.")

        return formulation_type.value(constraint_type=self.constraint_type, **formulation_kwargs)

    def sanity_check_multiplier(self, multiplier: MULTIPLIER_TYPE) -> None:

        if (self.constraint_type == ConstraintType.PENALTY) and not isinstance(
            multiplier, multipliers.ConstantMultiplier
        ):
            # If a penalty "constraint" is used, then we must have been provided a ConstantMultiplier.
            raise ValueError("A ConstantMultiplier must be provided along with a `penalty` constraint.")

        if isinstance(multiplier, multipliers.ConstantMultiplier):
            if any(multiplier() < 0) and (self.constraint_type == ConstraintType.INEQUALITY):
                raise ValueError("All entries of ConstantMultiplier must be non-negative for inequality constraints.")

        if isinstance(multiplier, multipliers.ExplicitMultiplier):
            if multiplier.implicit_constraint_type != self.constraint_type:
                raise ValueError(f"Provided multiplier is inconsistent with {self.constraint_type} constraint.")

    @property
    def state(self) -> ConstraintState:
        return self._state

    @state.setter
    def state(self, value: ConstraintState) -> None:
        if isinstance(self.multiplier, (multipliers.IndexedMultiplier, multipliers.ImplicitMultiplier)):
            if value.constraint_features is None:
                raise ValueError(f"Multipliers of type {type(self.multiplier)} expect constraint features.")

        self._state = value

    def compute_lagrangian_contribution(
        self, constraint_state: Optional[ConstraintState] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the contribution of the current constraint to the primal and dual
        Lagrangians, and evaluates the associated Lagrange multiplier."""

        if constraint_state is None and self.state is None:
            raise ValueError("A `ConstraintState` (provided or internal) is needed to compute Lagrangian contribution")
        elif constraint_state is None:
            constraint_state = self.state

        if constraint_state.constraint_features is None:
            multiplier_value = self.multiplier()
        else:
            multiplier_value = self.multiplier(constraint_state.constraint_features)

        if len(multiplier_value.shape) == 0:
            multiplier_value = multiplier_value.unsqueeze(0)

        primal_contribution, dual_contribution = self.formulation.compute_lagrangian_contribution(
            multiplier_value=multiplier_value, constraint_state=constraint_state
        )

        return multiplier_value, primal_contribution, dual_contribution

    def state_dict(self):
        return {
            "constraint_type": self.constraint_type,
            "formulation": self.formulation.state_dict(),
            "multiplier_state_dict": self.multiplier.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.constraint_type = state_dict["constraint_type"]
        self.formulation.load_state_dict(state_dict["formulation"])
        self.multiplier.load_state_dict(state_dict["multiplier_state_dict"])

    def __repr__(self):
        return f"ConstraintGroup(constraint_type={self.constraint_type}, formulation={self.formulation}, multiplier={self.multiplier})"


def observed_constraints_iterator(
    observed_constraints: Sequence[Union[ConstraintGroup, Tuple[ConstraintGroup, ConstraintState]]]
) -> Iterator[Tuple[ConstraintGroup, ConstraintState]]:
    """Utility function to iterate over observed constraints. This allows for consistent
    iteration over `observed_constraints` which are a sequence of `ConstraintGroup`\\s
    (and hold the `ConstraintState` internally), or a sequence of
    `Tuple[ConstraintGroup, ConstraintState]`\\s.
    """

    for constraint_tuple in observed_constraints:
        if isinstance(constraint_tuple, ConstraintGroup):
            constraint_group = constraint_tuple
            constraint_state = constraint_group.state
        elif isinstance(constraint_tuple, tuple) and len(constraint_tuple) == 2:
            constraint_group, constraint_state = constraint_tuple
        else:
            error_message = f"Received invalid format for observed constraint. Expected {ConstraintGroup} or"
            error_message += f" {Tuple[ConstraintGroup, ConstraintState]}, but received {type(constraint_tuple)}"
            raise ValueError(error_message)

        yield constraint_group, constraint_state
