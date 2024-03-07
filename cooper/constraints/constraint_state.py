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
        contributes_to_primal_update: When `False`, we ignore the contribution of the
            current observed constraint violation towards the primal Lagrangian, but
            keep their contribution to the dual Lagrangian. In other words, the observed
            violations affect the update for the dual variables but not the update for
            the primal variables.
        contributes_to_dual_update: When `False`, we ignore the contribution of the
            current observed constraint violation towards the dual Lagrangian, but keep
            their contribution to the primal Lagrangian. In other words, the observed
            violations affect the update for the primal variables but not the update
            for the dual variables. This flag is useful for performing less frequent
            updates of the dual variables (e.g. after several primal steps).
    """

    violation: torch.Tensor
    constraint_features: Optional[torch.Tensor] = None
    strict_violation: Optional[torch.Tensor] = None
    strict_constraint_features: Optional[torch.Tensor] = None
    contributes_to_primal_update: bool = True
    contributes_to_dual_update: bool = True

    def __post_init__(self):
        if self.constraint_features is not None and self.violation is None:
            raise ValueError("violation must be provided if constraint_features is provided.")

        if self.strict_constraint_features is not None and self.strict_violation is None:
            raise ValueError("strict_violation must be provided if strict_constraint_features is provided.")

    def extract_violations(self, do_unsqueeze=True) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts the violation and strict violation from the constraint state. If
        strict violations are not provided, patches them with the violation.
        This function also unsqueeze the violation tensors to ensure thay have at least
        1-dimension."""

        violation = self.violation

        if self.strict_violation is not None:
            strict_violation = self.strict_violation
        else:
            strict_violation = self.violation

        if do_unsqueeze:
            # If the violation is a scalar, we unsqueeze it to ensure that it has at
            # least one dimension for using einsum.
            if len(violation.shape) == 0:
                violation = violation.unsqueeze(0)
            if len(strict_violation.shape) == 0:
                strict_violation = strict_violation.unsqueeze(0)

        return violation, strict_violation

    def extract_constraint_features(self) -> torch.Tensor:
        """Extracts the constraint features from the constraint state.
        If strict constraint features are not provided, attempts to patch them with the
        differentiable constraint features. Similarly, if differentiable constraint
        features are not provided, attempts to patch them with the strict constraint
        features."""
        constraint_features = self.constraint_features

        if self.strict_constraint_features is not None:
            strict_constraint_features = self.strict_constraint_features
        else:
            strict_constraint_features = self.constraint_features

        return constraint_features, strict_constraint_features

    def compute_strictly_feasible_constraints(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the strictly feasible constraints and their features."""

        # FIXME(gallego-posada): This function is not being called
        # There is `ConstraintGroup > update_strictly_feasible_indices_` which is actually called
        # We should remove the current function.

        _, strict_violation = self.extract_violations()
        # When strict_violation is provided, we use it to determine the satisfaction of
        # the constraints. Otherwise, `extract_violations` patches it with the
        # differentiable violation.
        strictly_feasible_constraints = strict_violation < 0

        constraint_features = self.extract_constraint_features()

        return strictly_feasible_constraints, constraint_features


@dataclass
class ConstraintStore:
    # TODO: update docstring. Current ConstraintStore is agnostic to dual or primal
    # lagrangian.
    """Stores the value of the constraint factor (multiplier or penalty coefficient),
    the contribution of the constraint to the primal-differentiable Lagrian, and the
    contribution of the constraint to the dual-differentiable Lagrangian."""

    lagrangian_contribution: Optional[torch.Tensor] = None
    violation: Optional[torch.Tensor] = None
    multiplier_value: Optional[torch.Tensor] = None
    penalty_coefficient_value: Optional[torch.Tensor] = None
