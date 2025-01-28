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


def compute_dual_weighted_violation(
    multiplier_value: torch.Tensor, violation: torch.Tensor, penalty_coefficient_value: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""If a penalty coefficient is *not* provided, computes the sum of weighted constraint
    violations while preserving the gradient for the dual variables :math:`\vlambda` and
    :math:`\vmu` only. That is:

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
        penalty_coefficient_value: Tensor of penalty coefficient values.
    """
    args = [multiplier_value, violation.detach()]
    einsum_str = "i...,i..."

    if penalty_coefficient_value is not None:
        args.append(penalty_coefficient_value.detach())
        einsum_str += ",i..."

    return torch.einsum(f"{einsum_str}->", *args)


def compute_quadratic_penalty(
    penalty_coefficient_value: torch.Tensor, violation: torch.Tensor, constraint_type: ConstraintType
) -> Optional[torch.Tensor]:
    r"""A weighted sum of *squared* constraint violations using their associated penalty
    coefficients. This yields:

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
