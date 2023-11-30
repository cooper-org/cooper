import abc
import warnings
from typing import Optional

import torch

from cooper.constraints.constraint_state import ConstraintState, ConstraintType


class PenaltyCoefficient(torch.nn.Module, abc.ABC):
    """Abstract class for constant (non-trainable) coefficients used in penalized
    formulations.

    Args:
        init: Value of the penalty coefficient.
    """

    def __init__(self, init: torch.Tensor):
        if init.requires_grad:
            raise ValueError("PenaltyCoefficient should not require gradients.")
        self._value = init.clone()

    @property
    def value(self):
        """Return the current value of the penalty coefficient."""
        return self._value

    @value.setter
    def value(self, value: torch.Tensor):
        """Update the value of the penalty."""
        if value.requires_grad:
            raise ValueError("New value of PenaltyCoefficient should not require gradients.")
        if value.shape != self._value.shape:
            warnings.warn(
                f"New shape {value.shape} of PenaltyCoefficient does not match existing shape {self._value.shape}."
            )
        self._value = value.clone()

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """Move the penalty to a new device and/or change its dtype."""
        self.value = self.value.to(device=device, dtype=dtype)
        return self

    def state_dict(self):
        return {"value": self._value}

    def load_state_dict(self, state_dict):
        self._value = state_dict["value"]

    def __repr__(self):
        if self.value.numel() <= 10:
            return f"{type(self).__name__}({self.value})"
        else:
            return f"{type(self).__name__}(shape={self.value.shape})"

    @abc.abstractmethod
    def __call__(self):
        """Return the current value of the penalty coefficient."""
        pass


class DensePenaltyCoefficient(PenaltyCoefficient):
    """Constant (non-trainable) coefficient class used for penalized formulations."""

    def __call__(self):
        """Return the current value of the penalty coefficient."""
        return self.value

    def update_value(
        self,
        constraint_state: ConstraintState,
        constraint_type: ConstraintType,
        growth_factor: float,
        violation_tolerance: float,
    ):
        """Update the value of the penalty multiplicatively according to the provided
        growth factor. The penalty is only updated for violated constraints.
        """

        if violation_tolerance < 0.0:
            raise ValueError("Violation tolerance must be non-negative.")

        violation, strict_violation = constraint_state.extract_violations()

        if constraint_type == ConstraintType.INEQUALITY:
            new_value = torch.where(strict_violation > violation_tolerance, self._value * growth_factor, self._value)
        elif constraint_type == ConstraintType.EQUALITY:
            new_value = torch.where(
                strict_violation.abs() > violation_tolerance, self._value * growth_factor, self._value
            )

        self.value = new_value.detach()


class IndexedPenaltyCoefficient(PenaltyCoefficient):
    """Constant (non-trainable) coefficient class used in penalized formulations.
    When called, indexed penalty coefficients accept a tensor of indices and return the
    value of the penalty for a subset of constraints.
    """

    def __call__(self, indices: torch.Tensor):
        """Return the current value of the penalty coefficient at the provided indices.

        Args:
            indices: Tensor of indices for which to return the penalty coefficient.
        """

        if indices.dtype != torch.long:
            # Not allowing for boolean "indices", which are treated as indices by
            # torch.nn.functional.embedding and *not* as masks.
            raise ValueError("Indices must be of type torch.long.")

        coefficient_values = torch.nn.functional.embedding(indices, self._value, sparse=False)

        # Flatten coefficient values to 1D since Embedding works with 2D tensors.
        return torch.flatten(coefficient_values)

    def update_value(
        self,
        constraint_state: ConstraintState,
        constraint_type: ConstraintType,
        growth_factor: float,
        violation_tolerance: float,
    ):
        """Update the value of the penalty multiplicatively according to the provided
        growth factor. The penalty is only updated for violated constraints.
        """

        if violation_tolerance < 0.0:
            raise ValueError("Violation tolerance must be non-negative.")

        violation, strict_violation = constraint_state.extract_violations()
        constraint_features, strict_constraint_features = constraint_state.extract_constraint_features()

        values_for_observed = self._value[strict_constraint_features]

        if constraint_type == ConstraintType.INEQUALITY:
            new_value = torch.where(
                strict_violation > violation_tolerance, values_for_observed * growth_factor, values_for_observed
            )
        elif constraint_type == ConstraintType.EQUALITY:
            new_value = torch.where(
                strict_violation.abs() > violation_tolerance, values_for_observed * growth_factor, values_for_observed
            )

        self._value[strict_constraint_features] = new_value.detach()
