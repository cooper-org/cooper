from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch


class ConstraintType(Enum):
    EQUALITY = "equality"
    INEQUALITY = "inequality"
    PENALTY = "penalty"


@dataclass
class ConstraintState:
    """State of a constraint group describing the current constraint violation.

    Args:
        violation: Measurement of the constraint violation at some value of the primal
            parameters. This is expected to be differentiable with respect to the
            primal parameters.
        strict_violation: Measurement of the constraint violation which may be
            non-differentiable with respect to the primal parameters. When provided,
            the (necessarily differentiable) `violation` is used to compute the gradient
            of the Lagrangian with respect to the primal parameters, while the
            `strict_violation` is used to compute the gradient of the Lagrangian with
            respect to the dual parameters. For more details, see the proxy-constraint
            proposal of :cite:t:`cotter2019JMLR`.
        constraint_features: The features of the constraint. This is used to evaluate
            the lagrange multiplier associated with a constraint group. For example,
            An `IndexedMultiplier` expects the indices of the constraints whose Lagrange
            multipliers are to be retrieved; while an `ImplicitMultiplier` expects
            general tensor-valued features for the constraints. This field is not used
            for `DenseMultiplier`//s.
            This can be used in conjunction with an `IndexedMultiplier` to indicate the
            measurement of the violation for only a subset of the constraints within a
            `ConstraintGroup`.
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
    strict_violation: Optional[torch.Tensor] = None
    constraint_features: Optional[torch.Tensor] = None
    skip_primal_contribution: bool = False
    skip_dual_contribution: bool = False
