from typing import Optional, Tuple

import torch

from cooper.constraints.constraint_state import ConstraintType


def compute_primal_weighted_violation(
    constraint_factor_value: torch.Tensor, violation: torch.Tensor
) -> Optional[torch.Tensor]:
    """Computes the sum of constraint violations weighted by the associated constraint
    factors (multipliers or penalty coefficients), while preserving the gradient for the
    primal variables.

    Args:
        constraint_factor_value: The value of the multiplier or penalty coefficient for the
            constraint group.
        violation: Tensor of constraint violations.
    """

    if constraint_factor_value is None:
        raise ValueError("The constraint factor tensor must be provided if the primal contribution is not skipped.")
    if violation is None:
        raise ValueError("The violation tensor must be provided if the primal contribution is not skipped.")

    # When computing the gradient of the Lagrangian with respect to the primal
    # variables, we do not need to differentiate the multiplier. So we detach the
    # multiplier to avoid computing its gradient.
    # In the case of a penalty coefficient, the detach call is a no-op.
    return torch.einsum("i...,i...->", constraint_factor_value.detach(), violation)


def compute_dual_weighted_violation(
    constraint_factor_value: torch.Tensor,
    violation: torch.Tensor,
    penalty_coefficient_value: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Computes the sum of constraint violations weighted by the associated constraint
    factors (multipliers or penalty coefficients), while preserving the gradient for the
    dual variables.

    When computing the gradient of the Lagrangian with respect to the dual variables, we
    only need the _value_ of the constraint violation and not its gradient. So we detach
    the violation to avoid computing its gradient. Note that this enables the use of
    non-differentiable constraints for updating the multipliers.

    This insight was originally presented by Cotter et al. in the paper "Optimization
    with Non-Differentiable Constraints with Applications to Fairness, Recall, Churn,
    and Other Goals" under the name of "proxy" constraints.
    (https://jmlr.org/papers/v20/18-616.html, Sec. 4.2)

    Args:
        multiplier_value: The value of the multiplier for the constraint group.
        violation: Tensor of constraint violations.
    """

    if not constraint_factor_value.requires_grad:
        # If the constraint factor corresponds to a penalty coefficient, we can skip
        # the computation of the dual contribution since the penalty coefficient is not
        # trainable.
        return None
    else:
        multiplier_value = constraint_factor_value
        if multiplier_value is None:
            raise ValueError("The constraint factor tensor must be provided if the dual contribution is not skipped.")
        if violation is None:
            raise ValueError("The violation tensor must be provided if the dual contribution is not skipped.")

        detached_violation = violation.detach()

        if penalty_coefficient_value is None:
            return torch.einsum("i...,i...->", multiplier_value, detached_violation)
        else:
            # TODO(juan43ramirez): add comment explaining why the penalty coefficient is
            # used to re-weight the dual_lagrangian. This is motivated by Augmented
            # Lagrangian Method updates.
            return torch.einsum("i...,i...,i...->", multiplier_value, penalty_coefficient_value, detached_violation)


def compute_quadratic_penalty(
    penalty_coefficient_value: torch.Tensor,
    violation: torch.Tensor,
    strict_violation: torch.Tensor,
    constraint_type: ConstraintType,
) -> Optional[torch.Tensor]:
    # TODO(juan43ramirez): Add documentation

    if penalty_coefficient_value is None:
        raise ValueError("The penalty coefficient tensor must be provided if the primal contribution is not skipped.")
    if violation is None:
        raise ValueError("The violation tensor must be provided if the primal contribution is not skipped.")

    if constraint_type == ConstraintType.INEQUALITY:
        # We penalize the square violation associated with violated constraints.
        # This follows the setup from Eq. 17.7 in Numerical Optimization by
        # Nocedal and Wright (2006).

        # Violated constraintd are determined using the strict violation.
        constraint_filter = strict_violation >= 0

        sq_violation = constraint_filter * (violation**2)
    elif constraint_type == ConstraintType.EQUALITY:
        # Equality constraints do not need to be filtered
        sq_violation = violation**2
    else:
        # constraint_type not in [ConstraintType.INEQUALITY, ConstraintType.EQUALITY]:
        raise ValueError(f"{constraint_type} is incompatible with quadratic penalties.")

    return 0.5 * torch.einsum("i...,i...->", penalty_coefficient_value, sq_violation)
