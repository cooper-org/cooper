import math
from typing import Optional, Union

import torch

from cooper.multipliers import Multiplier
from cooper.penalty_coefficients import PenaltyCoefficient
from cooper.utils import ConstraintType


def evaluate_constraint_factor(
    module: Union[Multiplier, PenaltyCoefficient],
    constraint_features: Optional[torch.Tensor],
    expand_shape: tuple[int, ...],
) -> torch.Tensor:
    """Evaluate a Lagrange multiplier or penalty coefficient.

    If the module expects constraint features, it is called with the constraint features
    as an argument. Otherwise, it is called without arguments.

    Args:
        module: Multiplier or penalty coefficient module.
        constraint_features: The observed features of the constraint.
        expand_shape: Shape of the constraint violation tensor.
    """
    # TODO(gallego-posada): This way of calling the modules assumes either 0 or 1
    # arguments. This should be generalized to allow for multiple arguments.
    value = module(constraint_features) if module.expects_constraint_features else module()

    if value.dim() == 0:
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


def compute_primal_weighted_violation(
    constraint_factor_value: torch.Tensor, violation: torch.Tensor
) -> Optional[torch.Tensor]:
    r"""A weighted sum of constraint violations using their associated multipliers,
    preserving only the gradient for the primal variables :math:`\vx`. This corresponds
    to :math:`\vlambda .\texttt{detach}()^{\top} \vg(\vx)` for inequality constraints or
    :math:`\vmu .\texttt{detach}()^{\top} \vh(\vx)` for equality constraints.

    Args:
        constraint_factor_value: Tensor of constraint factor values.
        violation: Tensor of constraint violations.
    """
    # When computing the gradient of the Lagrangian with respect to the primal
    # variables, we do not need to differentiate the multiplier. So we detach the
    # multiplier to avoid computing its gradient.
    # In the case of a penalty coefficient, the detach call is a no-op.
    return torch.einsum("i...,i...->", constraint_factor_value.detach(), violation)


def compute_dual_weighted_violation(multiplier_value: torch.Tensor, violation: torch.Tensor) -> torch.Tensor:
    r"""Computes the sum of weighted constraint violations while preserving the gradient
    for the dual variables :math:`\vlambda` and :math:`\vmu` only.
    That is:

    .. math::
        \vlambda^{\top} \vg(\vx).\texttt{detach}() \text{ or } \vmu^{\top}
        \vh(\vx).\texttt{detach}()

    If a penalty coefficient is provided, the contribution of each violation is further
    multiplied by its associated penalty coefficient, ensuring that the gradient with
    respect to the multiplier is the constraint violation times the penalty coefficient.
    This results in:

    .. math::
        (\vlambda \odot \vc_{\vg})^{\top} \vg(\vx) \text{ or } (\vmu \odot
        \vc_{\vh})^{\top} \vh(\vx)


    Args:
        multiplier_value: Tensor of multiplier values.
        violation: Tensor of constraint violations.
    """
    return torch.einsum("i...,i...->", multiplier_value, violation.detach())


def compute_quadratic_penalty(
    penalty_coefficient_value: torch.Tensor, violation: torch.Tensor, constraint_type: ConstraintType
) -> Optional[torch.Tensor]:
    r"""A weighted sum of *squared* constraint violations using their associated penalty
    coefficients.

    We clamp the violations for inequality constraints as done in Eq 17.7 in Numerical
    Optimization by :cite:t:`nocedal2006NumericalOptimization`.
    This corresponds to:

    .. math::
        \frac{1}{2} \, \vc_{\vg}^{\top} \texttt{relu}(\vg(\vx))^2 \text{ or }
        \frac{1}{2} \, \vc_{\vh}^{\top} \vh(\vx)^2

    Args:
        penalty_coefficient_value: Tensor of penalty coefficient values.
        violation: Tensor of constraint violations.
        constraint_type: Type of constraint. One of ``ConstraintType.INEQUALITY`` or
            ``ConstraintType.EQUALITY``.

    """
    clamped_violation = torch.relu(violation) if constraint_type == ConstraintType.INEQUALITY else violation
    return 0.5 * torch.einsum("i...,i...->", penalty_coefficient_value, clamped_violation**2)


def compute_primal_quadratic_augmented_contribution(
    multiplier_value: torch.Tensor,
    penalty_coefficient_value: torch.Tensor,
    violation: torch.Tensor,
    constraint_type: ConstraintType,
) -> Optional[torch.Tensor]:
    r"""Computes the quadratic-augmented contribution of a constraint to the Lagrangian.

    When the constraint is an inequality constraint, the quadratic penalty is computed
    following Eqs 17.64 and 17.65 in Numerical Optimization by :cite:t:`nocedal2006NumericalOptimization`.
    Note that Nocedal and Wright use a "greater-than-or-equal to zero" convention for
    their constraints, which reverses some of the signs below. Denoting the current
    multiplier by :math:`\lambda` and the penalty coefficient by :math:`\rho`, we obtain
    the contribution of an inequality constraint to the augmented Lagrangian:

    .. math::
      \lambda_{*}^{\top} \text{violation} - \frac{1}{2 \rho} ||\lambda_{*} - \lambda||_2^2,

    where :math:`\lambda_{*}= \texttt{relu}(\lambda + \rho \text{violation})`. Note that
    this corresponds to the multiplier update after a step of projected gradient ascent.

    In the case of equality constraints, the quadratic-augmented contribution is computed
    following Eq 17.36 in :cite:t:`nocedal2006NumericalOptimization`:

    .. math::
        \lambda^{\top} \text{violation}+ \frac{rho}{2} ||violation||_2^2

    Args:
        multiplier_value: Tensor of multiplier values.
        penalty_coefficient_value: Tensor of penalty coefficient values.
        violation: Tensor of constraint violations.
        constraint_type: Type of constraint. One of `ConstraintType.INEQUALITY` or
            `ConstraintType.EQUALITY`.

    """
    if constraint_type == ConstraintType.INEQUALITY:
        aux1 = torch.einsum("i...,i...->i...", penalty_coefficient_value, violation)
        detached_multiplier = multiplier_value.detach()
        aux2 = torch.relu(detached_multiplier + aux1) ** 2 - detached_multiplier**2
        return 0.5 * torch.einsum("i...,i...->", 1 / penalty_coefficient_value, aux2)
    if constraint_type == ConstraintType.EQUALITY:
        linear_term = compute_primal_weighted_violation(multiplier_value, violation)
        quadratic_penalty = compute_quadratic_penalty(penalty_coefficient_value, violation, constraint_type)
        return linear_term + quadratic_penalty
    return None
