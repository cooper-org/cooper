from enum import Enum
from typing import Optional, Union

import torch

from cooper.constraints.constraint_state import ConstraintState

from .builders import build_explicit_multiplier
from .multipliers import DenseMultiplier, ExplicitMultiplier, ImplicitMultiplier, IndexedMultiplier, Multiplier
from .penalty_coefficients import DensePenaltyCoefficient, IndexedPenaltyCoefficient, PenaltyCoefficient

ConstraintFactor = Union[Multiplier, PenaltyCoefficient]


def evaluate_constraint_factor(module: ConstraintFactor, constraint_state: ConstraintState) -> torch.Tensor:
    """Evaluate the Lagrange multiplier or penalty coefficient associated with a
    constraint group.

    Args:
        module: Multiplier or penalty coefficient.
        constraint_state: The current state of the constraint.
    """
    value: torch.Tensor
    if constraint_state.constraint_features is None:
        value = module()
    else:
        value = module(constraint_state.constraint_features)

    if len(value.shape) == 0:
        value.unsqueeze_(0)

    if not value.requires_grad and value.numel() == 1 and constraint_state.violation.numel() > 1:
        # Expand the value of the penalty coefficient to match the shape of the violation.
        # This enables the use of a single penalty coefficient for all constraints in a
        # constraint group.
        # We only do this for penalty coefficients an not multipliers because we expect
        # a one-to-one mapping between multiplier values and constraints. If multiplier
        # sharing is desired, this should be done explicitly by the user.
        value = value.expand(constraint_state.violation.shape)

    return value


class MultiplierType(Enum):
    DENSE = DenseMultiplier
    INDEXED = IndexedMultiplier
    IMPLICIT = ImplicitMultiplier


class PenaltyCoefficientType(Enum):
    DENSE = DensePenaltyCoefficient
    INDEXED = IndexedPenaltyCoefficient
