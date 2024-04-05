from enum import Enum
from typing import Union

import torch

from .multipliers import DenseMultiplier, ExplicitMultiplier, ImplicitMultiplier, IndexedMultiplier, Multiplier
from .penalty_coefficients import DensePenaltyCoefficient, IndexedPenaltyCoefficient, PenaltyCoefficient

ConstraintFactor = Union[Multiplier, PenaltyCoefficient]


def evaluate_constraint_factor(
    module: ConstraintFactor, constraint_features: torch.Tensor, violation: torch.Tensor
) -> torch.Tensor:
    """Evaluate the Lagrange multiplier or penalty coefficient associated with a
    constraint.

    Args:
        module: Multiplier or penalty coefficient module.
        constraint_state: The current state of the constraint.
    """
    if violation is None:
        return None

    # TODO(gallego-posada): This way of calling the modules assumes either 0 or 1
    # arguments. This should be generalized to allow for multiple arguments.
    value = module() if constraint_features is None else module(constraint_features)

    if len(value.shape) == 0:
        value.unsqueeze_(0)

    if not value.requires_grad and value.numel() == 1 and violation.numel() > 1:
        # Expand the value of the penalty coefficient to match the shape of the violation.
        # This enables the use of a single penalty coefficient for all constraints in a
        # constraint.
        # We only do this for penalty coefficients an not multipliers because we expect
        # a one-to-one mapping between multiplier values and constraints. If multiplier
        # sharing is desired, this should be done explicitly by the user.
        value = value.expand(violation.shape)

    return value


class MultiplierType(Enum):
    DENSE = DenseMultiplier
    INDEXED = IndexedMultiplier
    IMPLICIT = ImplicitMultiplier


class PenaltyCoefficientType(Enum):
    DENSE = DensePenaltyCoefficient
    INDEXED = IndexedPenaltyCoefficient
