import math
from enum import Enum
from typing import Union

import torch

from .multipliers import DenseMultiplier, ExplicitMultiplier, ImplicitMultiplier, IndexedMultiplier, Multiplier
from .penalty_coefficients import DensePenaltyCoefficient, IndexedPenaltyCoefficient, PenaltyCoefficient

ConstraintFactor = Union[Multiplier, PenaltyCoefficient]


def evaluate_constraint_factor(
    module: ConstraintFactor, constraint_features: torch.Tensor, expand_shape: torch.Tensor
) -> torch.Tensor:
    """Evaluate the Lagrange multiplier or penalty coefficient associated with a
    constraint.

    Args:
        module: Multiplier or penalty coefficient module.
        constraint_state: The current state of the constraint.
    """

    # TODO(gallego-posada): This way of calling the modules assumes either 0 or 1
    # arguments. This should be generalized to allow for multiple arguments.
    value = module() if constraint_features is None else module(constraint_features)

    if len(value.shape) == 0:
        # Unsqueeze value to make it a 1D tensor for consistent use in Formulations' einsum  calls
        value.unsqueeze_(0)

    if not value.requires_grad and value.numel() == 1 and math.prod(expand_shape) > 1:
        # Expand the value of the penalty coefficient to match the shape of the violation.
        # This enables the use of a single penalty coefficient for all constraints in a
        # constraint.
        # We only do this for penalty coefficients an not multipliers (note the
        # `requires_grad` check) because we expect a one-to-one mapping between
        # multiplier values and constraint violation values. If multiplier sharing is
        # desired, the user must implement this explicitly.
        value = value.expand(expand_shape)

    return value


class MultiplierType(Enum):
    DENSE = DenseMultiplier
    INDEXED = IndexedMultiplier
    IMPLICIT = ImplicitMultiplier


class PenaltyCoefficientType(Enum):
    DENSE = DensePenaltyCoefficient
    INDEXED = IndexedPenaltyCoefficient
