from enum import Enum
from typing import Optional, Union

import torch

from cooper.constraints.constraint_state import ConstraintState

from .builders import build_explicit_multiplier
from .multipliers import DenseMultiplier, ExplicitMultiplier, ImplicitMultiplier, IndexedMultiplier, Multiplier
from .penalty_coefficients import DensePenaltyCoefficient, IndexedPenaltyCoefficient, PenaltyCoefficient

ConstraintFactor = Union[Multiplier, PenaltyCoefficient]


def evaluate_constraint_factor(
    module: ConstraintFactor, violation: torch.Tensor, constraint_features: torch.Tensor
) -> torch.Tensor:
    """Evaluate the Lagrange multiplier or penalty coefficient associated with a
    constraint group.

    Args:
        module: Multiplier or penalty coefficient.
        constraint_state: The current state of the constraint.
    """
    value: torch.Tensor

    if violation is None:
        return None

    if constraint_features is None:
        value = module()
    else:
        value = module(constraint_features)

    if len(value.shape) == 0:
        value.unsqueeze_(0)

    if not value.requires_grad and value.numel() == 1 and violation.numel() > 1:
        # Expand the value of the penalty coefficient to match the shape of the violation.
        # This enables the use of a single penalty coefficient for all constraints in a
        # constraint group.
        # We only do this for penalty coefficients an not multipliers because we expect
        # a one-to-one mapping between multiplier values and constraints. If multiplier
        # sharing is desired, this should be done explicitly by the user.
        value = value.expand(violation.shape)

    return value


def evaluate_constraint_factor_for_primal_and_dual(
    module: ConstraintFactor, constraint_state: ConstraintState
) -> torch.Tensor:
    multiplier_value = evaluate_constraint_factor(
        module, constraint_state.violation, constraint_state.constraint_features
    )

    if constraint_state.strict_constraint_features is None:
        strict_constraint_features = constraint_state.constraint_features
    else:
        strict_constraint_features = constraint_state.strict_constraint_features

    strict_multiplier_value = evaluate_constraint_factor(
        module, constraint_state.strict_violation, strict_constraint_features
    )

    if strict_multiplier_value is None:
        strict_multiplier_value = multiplier_value

    return multiplier_value, strict_multiplier_value


class MultiplierType(Enum):
    DENSE = DenseMultiplier
    INDEXED = IndexedMultiplier
    IMPLICIT = ImplicitMultiplier


class PenaltyCoefficientType(Enum):
    DENSE = DensePenaltyCoefficient
    INDEXED = IndexedPenaltyCoefficient
