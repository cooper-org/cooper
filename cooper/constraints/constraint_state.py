from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import torch


class ConstraintType(Enum):
    EQUALITY = auto()
    INEQUALITY = auto()
    PENALTY = auto()


@dataclass
class ConstraintState:
    """State of a constraint group describing the current constraint violation.

    Args:
        violation: Measurement of the constraint violation at some value of the primal
            parameters. This is expected to be differentiable with respect to the
            primal parameters.
        constraint_features: The features of the (differentiable) constraint. This is
            used to evaluate the Lagrange multiplier associated with a constraint group.
            For example, an `IndexedMultiplier` expects the indices of the constraints
            whose Lagrange multipliers are to be retrieved; while an
            `ImplicitMultiplier` expects general tensor-valued features for the
            constraints. This field is not used for `DenseMultiplier`//s.
            This can be used in conjunction with an `IndexedMultiplier` to indicate the
            measurement of the violation for only a subset of the constraints within a
            `ConstraintGroup`.
        strict_violation: Measurement of the constraint violation which may be
            non-differentiable with respect to the primal parameters. When provided,
            the (necessarily differentiable) `violation` is used to compute the gradient
            of the Lagrangian with respect to the primal parameters, while the
            `strict_violation` is used to compute the gradient of the Lagrangian with
            respect to the dual parameters. For more details, see the proxy-constraint
            proposal of :cite:t:`cotter2019JMLR`.
        strict_constraint_features: The features of the (possibly non-differentiable)
            constraint. For more details, see `constraint_features`.
        skip_primal_conribution: When `True`, we ignore the contribution of the current
            observed constraint violation towards the primal Lagrangian, but keep their
            contribution to the dual Lagrangian. In other words, the observed violations
            affect the update for the dual variables but not the update for the primal
            variables.
        skip_dual_conribution: When `True`, we ignore the contribution of the current
            observed constraint violation towards the dual Lagrangian, but keep their
            contribution to the primal Lagrangian. In other words, the observed
            violations affect the update for the primal variables but not the update
            for the dual variables. This flag is useful for performing less frequent
            updates of the dual variables (e.g. after several primal steps).
    """

    violation: torch.Tensor
    constraint_features: Optional[torch.Tensor] = None
    strict_violation: Optional[torch.Tensor] = None
    strict_constraint_features: Optional[torch.Tensor] = None
    # TODO: use_in_primal_update: bool = True, use_in_dual_update: bool = True
    skip_primal_contribution: bool = False
    skip_dual_contribution: bool = False


@dataclass
class ConstraintStore:
    """Stores the value of the constraint factor (multiplier or penalty coefficient),
    the contribution of the constraint to the primal-differentiable Lagrian, and the
    contribution of the constraint to the dual-differentiable Lagrangian."""

    lagrangian_contribution: Optional[torch.Tensor] = None
    violation: Optional[torch.Tensor] = None
    multiplier_value: Optional[torch.Tensor] = None
    penalty_coefficient_value: Optional[torch.Tensor] = None
