import abc
import warnings
from typing import Optional, Sequence

import torch

from cooper.constraints import Constraint, ConstraintState, ConstraintType


class PenaltyCoefficient(abc.ABC):
    """Abstract class for constant (non-trainable) coefficients used in Augmented Lagrangian formulation.

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
    def __call__(self, *args, **kwargs):
        """Return the current value of the penalty coefficient."""
        pass


class DensePenaltyCoefficient(PenaltyCoefficient):
    """Constant (non-trainable) coefficient class used for Augmented Lagrangian formulation."""

    @torch.inference_mode()
    def __call__(self):
        """Return the current value of the penalty coefficient."""
        return self.value.clone()


class IndexedPenaltyCoefficient(PenaltyCoefficient):
    """Constant (non-trainable) coefficient class used in Augmented Lagrangian formulation.
    When called, indexed penalty coefficients accept a tensor of indices and return the
    value of the penalty for a subset of constraints.
    """

    @torch.inference_mode()
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


class PenaltyCoefficientUpdater(abc.ABC):
    """Abstract class for updating the penalty coefficient of a constraint."""

    def step(self, observed_constraints: Sequence[tuple[Constraint, ConstraintState]]):
        for constraint, constraint_state in observed_constraints:
            # If a constraint does not contribute to the dual update, we do not update
            # its penalty coefficient.
            if constraint_state.contributes_to_dual_update:
                self.update_penalty_coefficient_(constraint, constraint_state)

    @abc.abstractmethod
    def update_penalty_coefficient_(self, constraint: Constraint, constraint_state: ConstraintState) -> None:
        pass


class MultiplicativePenaltyCoefficientUpdater(PenaltyCoefficientUpdater):
    """Multiplicative penalty coefficient updater for Augmented Lagrangian formulation.
    The penalty coefficient is updated by multiplying it by a growth factor when the constraint
    violation is larger than a given tolerance.
    Based on Algorithm 17.4 in Numerical Optimization by Nocedal and Wright.

    Args:
        growth_factor: The factor by which the penalty coefficient is multiplied when the
            constraint is violated beyond ``violation_tolerance``.
        violation_tolerance: The tolerance for the constraint violation. If the violation
            is smaller than this tolerance, the penalty coefficient is not updated.
            The comparison is done at the constraint-level (i.e., each entry of the
            violation tensor). For equality constraints, the absolute violation is
            compared to the tolerance. All constraint types use the strict violation
            (when available) for the comparison.
    """

    def __init__(self, growth_factor: float = 1.01, violation_tolerance: float = 1e-4):
        if violation_tolerance < 0.0:
            raise ValueError("Violation tolerance must be non-negative.")

        self.growth_factor = growth_factor
        self.violation_tolerance = violation_tolerance

    def update_penalty_coefficient_(self, constraint: Constraint, constraint_state: ConstraintState) -> None:

        violation, strict_violation = constraint_state.extract_violations()
        constraint_features, strict_constraint_features = constraint_state.extract_constraint_features()
        penalty_coefficient = constraint.penalty_coefficient

        values_for_observed = (
            penalty_coefficient.value
            if isinstance(penalty_coefficient, DensePenaltyCoefficient)
            else penalty_coefficient.value[strict_constraint_features]
        )

        if constraint.constraint_type == ConstraintType.EQUALITY:
            condition = strict_violation.abs() > self.violation_tolerance
        else:
            condition = strict_violation > self.violation_tolerance

        new_value = torch.where(condition, values_for_observed * self.growth_factor, values_for_observed)

        if isinstance(penalty_coefficient, DensePenaltyCoefficient):
            penalty_coefficient.value = new_value.detach()
        else:
            penalty_coefficient.value[strict_constraint_features] = new_value.detach()
