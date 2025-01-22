import abc
from typing import Any, Optional

import torch
from typing_extensions import Self


class PenaltyCoefficient(abc.ABC):
    """Abstract class for constant (non-trainable) coefficients used in Augmented Lagrangian formulation.

    Args:
        init: Value of the penalty coefficient.
    """

    expects_constraint_features: bool
    _value: Optional[torch.Tensor] = None

    def __init__(self, init: torch.Tensor) -> None:
        if init.dim() > 1:
            raise ValueError("init must either be a scalar or a 1D tensor of shape `(num_constraints,)`.")
        self.value = init

    @property
    def value(self) -> torch.Tensor:
        """Return the current value of the penalty coefficient."""
        return self._value

    @value.setter
    def value(self, value: torch.Tensor) -> None:
        """Update the value of the penalty."""
        if value.requires_grad:
            raise ValueError("PenaltyCoefficient should not require gradients.")
        if self._value is not None and value.shape != self._value.shape:
            raise ValueError(
                f"New shape {value.shape} of PenaltyCoefficient does not match existing shape {self._value.shape}."
            )
        self._value = value.clone()
        self.sanity_check()

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> Self:
        """Move the penalty coefficient to a new device and/or change its dtype.

        Args:
            device: The desired device of the penalty coefficient.
            dtype: The desired dtype of the penalty coefficient.
        """
        self._value = self._value.to(device=device, dtype=dtype)
        return self

    def state_dict(self) -> dict:
        """Return the current state of the penalty coefficient."""
        return {"value": self._value}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state of the penalty coefficient.

        Args:
            state_dict: Dictionary containing the state of the penalty coefficient.
        """
        self._value = state_dict["value"]

    def sanity_check(self) -> None:
        if torch.any(self._value < 0):
            raise ValueError("All entries of the penalty coefficient must be non-negative.")

    def __repr__(self) -> str:
        if self.value.numel() <= 10:
            return f"{type(self).__name__}({self.value})"
        return f"{type(self).__name__}(shape={self.value.shape})"

    @abc.abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Return the current value of the penalty coefficient."""


class DensePenaltyCoefficient(PenaltyCoefficient):
    """Constant (non-trainable) coefficient class used for Augmented Lagrangian formulation."""

    expects_constraint_features = False

    @torch.no_grad()
    def __call__(self) -> torch.Tensor:
        """Return the current value of the penalty coefficient."""
        return self.value.clone()


class IndexedPenaltyCoefficient(PenaltyCoefficient):
    """Constant (non-trainable) coefficient class used in Augmented Lagrangian formulation.
    When called, indexed penalty coefficients accept a tensor of indices and return the
    value of the penalty for a subset of constraints.
    """

    expects_constraint_features = True

    @torch.no_grad()
    def __call__(self, indices: torch.Tensor) -> torch.Tensor:
        """Return the current value of the penalty coefficient at the provided indices.

        Args:
            indices: Tensor of indices for which to return the penalty coefficient.
        """
        if indices.dtype != torch.long:
            # Not allowing for boolean "indices", which are treated as indices by
            # torch.nn.functional.embedding and *not* as masks.
            raise ValueError("Indices must be of type torch.long.")

        if self.value.dim() == 0:
            return self.value.clone()

        coefficient_values = torch.nn.functional.embedding(indices, self.value.unsqueeze(1), sparse=False)

        # Flatten coefficient values to 1D since Embedding works with 2D tensors.
        return torch.flatten(coefficient_values)
