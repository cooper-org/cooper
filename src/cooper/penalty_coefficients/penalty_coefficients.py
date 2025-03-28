import abc
from typing import Any, Optional

import torch
from typing_extensions import Self


class PenaltyCoefficient(abc.ABC):
    """Abstract class for constant (non-trainable) penalty coefficients.

    Args:
        init: Value of the penalty coefficient.

    Raises:
        ValueError: If ``init`` has two or more dimensions.
    """

    expects_constraint_features: bool
    _value: Optional[torch.Tensor] = None

    def __init__(self, init: torch.Tensor) -> None:
        if init.dim() > 1:
            raise ValueError("init must either be a scalar or a 1D tensor of shape `(num_constraints,)`.")
        self.init = init.clone()
        self.value = init

    @property
    def value(self) -> torch.Tensor:
        """Return the current value of the penalty coefficient."""
        return self._value

    @value.setter
    def value(self, value: torch.Tensor) -> None:
        """Update the value of the penalty.

        Raises:
            ValueError: if the provided ``value`` has a different shape than the
                existing one or contains negative entries.

        """
        if value.requires_grad:
            raise ValueError("PenaltyCoefficient should not require gradients.")
        if self._value is not None and value.shape != self._value.shape:
            raise ValueError(
                f"New shape {value.shape} of PenaltyCoefficient does not match existing shape {self._value.shape}."
            )
        self._value = value.clone()
        self.sanity_check()

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Move the penalty coefficient to a new ``device`` and/or change its
        ``dtype``.
        """
        self._value = self._value.to(*args, **kwargs)
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
        """Check that the penalty coefficient is well-formed.

        Raises:
            ValueError: If the penalty coefficient contains negative entries.
        """
        if torch.any(self._value < 0):
            raise ValueError("All entries of the penalty coefficient must be non-negative.")

    def __repr__(self) -> str:
        if self.value.numel() <= 10:  # noqa: PLR2004
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
    """Constant (non-trainable) penalty coefficients. When called, indexed penalty
    coefficients accept a tensor of indices and return the value of the penalty for
    a subset of constraints.
    """

    expects_constraint_features = True

    @torch.no_grad()
    def __call__(self, indices: torch.Tensor) -> torch.Tensor:
        """Return the current value of the penalty coefficient at the provided indices.

        Args:
            indices: Tensor of indices for which to return the penalty coefficient.

        Raises:
            ValueError: If ``indices`` is not of type ``torch.long``.
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
