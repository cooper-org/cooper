import warnings
from dataclasses import dataclass
from typing import Iterator, Literal, Optional, Sequence, Tuple, Union

import torch

from cooper import multipliers
from cooper.formulation import FORMULATION_TYPE, Formulation

CONSTRAINT_TYPE = Literal["eq", "ineq", "penalty"]


@dataclass
class ConstraintState:
    """Constraint state."""

    # TODO(gallego-posada): Add documentation

    violation: torch.Tensor
    strict_violation: Optional[torch.Tensor] = None
    constraint_features: Optional[torch.Tensor] = None
    skip_primal_contribution: bool = False
    skip_dual_contribution: bool = False


class ConstraintGroup:
    """Constraint Group."""

    # TODO(gallego-posada): Add documentation

    def __init__(
        self,
        constraint_type: CONSTRAINT_TYPE,
        formulation_type: Optional[FORMULATION_TYPE] = "lagrangian",
        formulation_kwargs: Optional[dict] = {},
        multiplier: Optional[multipliers.MULTIPLIER_TYPE] = None,
        shape: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        is_sparse: Optional[bool] = False,
    ):

        self._state: ConstraintState = None

        self.constraint_type = constraint_type
        self.formulation = self.build_formulation(formulation_type, formulation_kwargs)

        if multiplier is None:
            multiplier = self.build_multiplier(shape=shape, dtype=dtype, device=device, is_sparse=is_sparse)
        self.sanity_check_multiplier(multiplier)
        self.multiplier = multiplier

    def build_formulation(self, formulation_type, formulation_kwargs):
        if self.constraint_type == "penalty":
            # `penalty` constraints must be paired with "penalized" formulations. If no formulation is provided, we
            # default to a "penalized" formulation.
            if formulation_type != "penalized":
                warning_message = (
                    "A constraint of type `penalty` must be used with a `penalized` formulation, but received"
                    f" formulation_type={formulation_type}. The formulation type will be set to `penalized`."
                    " Please review your configuration and override the default formulation_type='lagrangian'."
                )
                warnings.warn(warning_message)
            formulation_type = "penalized"

        return Formulation(formulation_type, **formulation_kwargs)

    def sanity_check_multiplier(self, multiplier: multipliers.MULTIPLIER_TYPE) -> None:

        if (self.constraint_type == "penalty") and not isinstance(multiplier, multipliers.ConstantMultiplier):
            # If a penalty "constraint" is used, then we must have been provided a ConstantMultiplier.
            raise ValueError("A ConstantMultiplier must be provided along with a `penalty` constraint.")

        if isinstance(multiplier, multipliers.ConstantMultiplier):
            if any(multiplier() < 0) and (self.constraint_type == "ineq"):
                raise ValueError("All entries of ConstantMultiplier must be non-negative for inequality constraints.")

        if isinstance(multiplier, multipliers.ExplicitMultiplier):
            if multiplier.implicit_constraint_type != self.constraint_type:
                raise ValueError(f"Provided multiplier is inconsistent with {self.constraint_type} constraint.")

    @property
    def state(self) -> ConstraintState:
        return self._state

    @state.setter
    def state(self, value: ConstraintState) -> None:
        if isinstance(self.multiplier, (multipliers.SparseMultiplier, multipliers.ImplicitMultiplier)):
            if value.constraint_features is None:
                raise ValueError(f"Multipliers of type {type(self.multiplier)} expect constraint features.")

        self._state = value

    def build_multiplier(self, shape: int, dtype: torch.dtype, device: torch.device, is_sparse: bool) -> None:

        multiplier_class = multipliers.SparseMultiplier if is_sparse else multipliers.DenseMultiplier

        tensor_factory = dict(dtype=dtype, device=device)
        tensor_factory["size"] = (shape, 1) if is_sparse else (shape,)

        return multiplier_class(init=torch.zeros(**tensor_factory), enforce_positive=(self.constraint_type == "ineq"))

    def compute_lagrangian_contribution(
        self, constraint_state: Optional[ConstraintState] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if constraint_state is None and self.state is None:
            raise ValueError("A `ConstraintState` (provided or internal) is needed to compute Lagrangian contribution")
        elif constraint_state is None:
            constraint_state = self.state

        if constraint_state.constraint_features is None:
            multiplier_value = self.multiplier()
        else:
            multiplier_value = self.multiplier(constraint_state.constraint_features)

        # Strict violation represents the "actual" violation of the constraint.
        # We use it to update the value of the multiplier.
        if constraint_state.strict_violation is not None:
            strict_violation = constraint_state.strict_violation
        else:
            # If strict violation is not provided, we use the differentiable
            # violation (which always exists).
            strict_violation = constraint_state.violation

        primal_contribution, dual_contribution = self.formulation.compute_lagrangian_contribution(
            self.constraint_type, multiplier_value, constraint_state.violation, strict_violation
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
