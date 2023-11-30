from typing import Optional

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

    When both the constraint factor and the penalty coefficient are provided, the
    contribution for each violation is weighted by both the value of the multiplier and
    the penalty coefficient. This way, the gradient with respect to the multiplier is
    the constraint violation times the penalty coefficient, as required by the updates
    of the Augmented Lagrangian Method. See Eq. 5.62 in Nonlinear Programming by
    Bertsekas (2016).

    Args:
        multiplier_value: The value of the multiplier for the constraint group.
        violation: Tensor of constraint violations.
        penalty_coefficient_value: Tensor of penalty coefficient values.
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
            return torch.einsum(
                "i...,i...,i...->", multiplier_value, penalty_coefficient_value.detach(), detached_violation
            )


def compute_quadratic_augmented_contribution(
    multiplier_value: torch.Tensor,
    penalty_coefficient_value: torch.Tensor,
    violation: torch.Tensor,
    constraint_type: ConstraintType,
) -> Optional[torch.Tensor]:
    r"""
    Computes the quadratic penalty for a constraint group.

    When the constraint is an inequality constraint, the quadratic penalty is computed
    following Eq 17.65 in Numerical Optimization by Nocedal and Wright (2006). Denoting
    the multiplier by :math:`\lambda` and the penalty coefficient by :math:`\rho`, the

    .. math::
      \frac{1}{2 \rho} ( || max(0, \lambda + violation * rho) ||_2^2 - || \lambda ||_2^2)

    Note that when the multiplier is zero, the formula simplifies to the standard
    quadratic penalty for inequality constraints.
    .. math::
        \frac{\rho}{2} || max(0, violation) ||_2^2

    When the constraint is an equality constraint, the quadratic penalty is computed
    following Eq 17.36 in Numerical Optimization by Nocedal and Wright (2006). Note
    that, unlike inequality constraints, there is no thresholding at zero for equality
    constraints.

    .. math::
        \frac{rho}{2} ||violation||_2^2

    """

    if penalty_coefficient_value is None:
        raise ValueError("The penalty coefficient tensor must be provided if the primal contribution is not skipped.")
    if violation is None:
        raise ValueError("The violation tensor must be provided if the primal contribution is not skipped.")

    if constraint_type == ConstraintType.INEQUALITY:

        aux1 = torch.einsum("i...,i...->i...", penalty_coefficient_value, violation)

        if multiplier_value is None:
            return 0.5 * torch.einsum("i...,i...->", 1 / penalty_coefficient_value, torch.relu(aux1) ** 2)
        else:
            aux2 = torch.relu(multiplier_value + aux1) ** 2 - multiplier_value**2
            return 0.5 * torch.einsum("i...,i...->", 1 / penalty_coefficient_value, aux2)

    elif constraint_type == ConstraintType.EQUALITY:
        if multiplier_value is None:
            linear_contribution = 0.0
        else:
            linear_contribution = compute_primal_weighted_violation(
                constraint_factor_value=multiplier_value, violation=violation
            )
        quadratic_penalty = 0.5 * torch.einsum("i...,i...->", penalty_coefficient_value, violation**2)
        return linear_contribution + quadratic_penalty
    else:
        # constraint_type not in [ConstraintType.INEQUALITY, ConstraintType.EQUALITY]:
        raise ValueError(f"{constraint_type} is incompatible with quadratic penalties.")
