import functools
from enum import Enum
from typing import Literal, Optional, Tuple

import torch

from cooper.constraints.constraint_state import ConstraintState, ConstraintType


def extract_and_patch_violations(constraint_state: ConstraintState):
    """Extracts the violation and strict violation from the constraint state, and
    patches the strict violation if it is not available. We also unsqueeze the
    violation tensors to ensure thay have at least 1-dimension."""

    violation = constraint_state.violation
    if len(violation.shape) == 0:
        violation = violation.unsqueeze(0)

    strict_violation = constraint_state.strict_violation
    if strict_violation is None:
        strict_violation = constraint_state.violation
    if len(strict_violation.shape) == 0:
        strict_violation = strict_violation.unsqueeze(0)

    return violation, strict_violation


def compute_primal_weighted_violation(multiplier_value: torch.Tensor, constraint_state) -> Optional[torch.Tensor]:
    """Computes the sum of constraint violations weighted by the associated multipliers,
    while preserving the gradient for the primal variables.

    Args:
        multiplier_value: The value of the multiplier for the constraint group.
        constraint_state: The current state of the constraint.
    """

    if constraint_state.skip_primal_contribution:
        # Ignore the primal contribution if the constraint is marked as non-contributing
        # to the primal Lagrangian.
        return None
    else:
        violation, _ = extract_and_patch_violations(constraint_state)

        if multiplier_value is None:
            raise ValueError("The multiplier tensor must be provided if the primal contribution is not skipped.")
        if violation is None:
            raise ValueError("The violation tensor must be provided if the primal contribution is not skipped.")

        # When computing the gradient of the Lagrangian with respect to the primal
        # variables, we do not need to differentiate the multiplier.
        return torch.einsum("i...,i...->", multiplier_value.detach(), violation)


def compute_dual_weighted_violation(multiplier_value: torch.Tensor, constraint_state) -> Optional[torch.Tensor]:
    """Computes the sum of constraint violations weighted by the associated multipliers,
    while preserving the gradient for the dual variables.

    When computing the gradient of the Lagrangian with respect to the dual variables, we
    only need the _value_ of the constraint violation and not its gradient. So we detach
    the violation to avoid computing its gradient. Note that this enables the use of
    non-differentiable constraints for updating the multiplier.

    This insight was originally proposed by Cotter et al. in the paper "Optimization
    with Non-Differentiable Constraints with Applications to Fairness, Recall, Churn,
    and Other Goals" under the name of "proxy" constraints.
    (https://jmlr.org/papers/v20/18-616.html, Sec. 4.2)

    Args:
        multiplier_value: The value of the multiplier for the constraint group.
        constraint_state: The current state of the constraint.
    """

    if constraint_state.skip_dual_contribution:
        # Ignore the primal contribution if the constraint is marked as non-contributing
        # to the dual Lagrangian.
        return None
    else:
        if multiplier_value is None:
            raise ValueError("The multiplier tensor must be provided if the primal contribution is not skipped.")

        # Strict violation represents the "actual" violation of the constraint. When
        # provided, we use the strict violation to update the value of the multiplier.
        # Otherwise, we default to using the differentiable violation.
        _, strict_violation = extract_and_patch_violations(constraint_state)

        return torch.einsum("i...,i...->", multiplier_value, strict_violation.detach())


class Formulation:
    def __init__(self, formulation_type: Literal["penalized", "lagrangian"], constraint_type: ConstraintType):
        # TODO(gallego-posada): Add documentation
        self.formulation_type = formulation_type
        self.constraint_type = constraint_type

    def compute_lagrangian_contribution(
        self, multiplier_value: torch.Tensor, constraint_state: ConstraintState
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

        weighted_violation_for_primal = compute_primal_weighted_violation(multiplier_value, constraint_state)

        if self.formulation_type == "penalized":
            # Penalized formulations have no _trainable_ dual variables, so we adopt the
            # convention of setting this variable to None.
            weighted_violation_for_dual = None
        else:
            weighted_violation_for_dual = compute_dual_weighted_violation(multiplier_value, constraint_state)

        return weighted_violation_for_primal, weighted_violation_for_dual

    def state_dict(self):
        return {"formulation_type": self.formulation_type}

    def load_state_dict(self, state_dict):
        self.formulation_type = state_dict["formulation_type"]

    def __repr__(self):
        return f"{self.formulation_type.capitalize()}Formulation"


def compute_primal_augmented_lagrangian_penalty(
    constraint_type: ConstraintType,
    multiplier_value: torch.Tensor,
    constraint_state: ConstraintState,
    augmented_lagrangian_scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> Optional[float | torch.Tensor]:
    """Computes the augmented Lagrangian penalty, while preserving the gradient for the
    primal variables.

    Args:
        constraint_type: The type of constraint. Used to determine whether the
            constraint should be filtered based on it feasibility.
        multiplier_value: The value of the multiplier for the constraint group.
        constraint_state: The current state of the constraint.
    """

    if constraint_state.skip_primal_contribution:
        return None

    augmented_lagrangian_penalty = 0.0

    violation, strict_violation = extract_and_patch_violations(constraint_state)

    if constraint_type == ConstraintType.INEQUALITY:
        # Compute filter based on strict constraint violation
        const_filter = torch.logical_or(strict_violation >= 0, multiplier_value > 0)
        sq_violation = torch.sum(const_filter.detach() * (violation**2))
    elif constraint_type == ConstraintType.EQUALITY:
        # Equality constraints do not need to be filtered
        sq_violation = torch.sum(violation**2)
    else:
        raise ValueError(f"{constraint_type} is incompatible with formulation_type=augmented_lagrangian")

    # TODO(gallego-posada): Why were we doing this check before?
    # # Gather all the learning rates for the "parameter groups" of the dual
    # # variables, and check that all the learning rates are the same.
    dual_lrs = augmented_lagrangian_scheduler.get_last_lr()
    # is_all_dual_lr_equal = all(x == dual_lrs[0] for x in dual_lrs)
    # assert is_all_dual_lr_equal, "All the dual LRs must be the same."

    # Use the dual learning as the Augmented Lagrangian coefficient to
    # ensure that gradient-based update will coincide with the update
    # scheme of the Augmented Lagrangian method.
    augmented_lagrangian_coefficient = dual_lrs[0]
    if augmented_lagrangian_coefficient > 0:
        # If using augmented Lagrangian, add squared sum of constraints
        # Following the formulation on Marc Toussaint slides (p 17-20)
        # https://ipvs.informatik.uni-stuttgart.de/mlr/marc/teaching/13-Optimization/03-constrainedOpt.pdf
        augmented_lagrangian_penalty += 0.5 * augmented_lagrangian_coefficient * sq_violation

    return augmented_lagrangian_penalty


class AugmentedLagrangianFormulation:
    def __init__(
        self, constraint_type: ConstraintType, augmented_lagrangian_scheduler: torch.optim.lr_scheduler._LRScheduler
    ):
        self.formulation_type = "augmented_lagrangian"
        self.constraint_type = constraint_type
        self.augmented_lagrangian_scheduler = augmented_lagrangian_scheduler

    def compute_lagrangian_contribution(
        self, multiplier_value: torch.Tensor, constraint_state: ConstraintState
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

        raise NotImplementedError("This formulation is not yet tested.")

        weighted_violation_for_primal = compute_primal_weighted_violation(multiplier_value, constraint_state)
        weighted_violation_for_dual = compute_dual_weighted_violation(multiplier_value, constraint_state)

        augmented_lagrangian_penalty_for_primal = compute_primal_augmented_lagrangian_penalty(
            self.constraint_type, multiplier_value, constraint_state, self.augmented_lagrangian_scheduler
        )
        assert (weighted_violation_for_primal is None) == (augmented_lagrangian_penalty_for_primal is None)

        if (weighted_violation_for_primal is not None) and (augmented_lagrangian_penalty_for_primal is None):
            weighted_violation_for_primal += augmented_lagrangian_penalty_for_primal

        return weighted_violation_for_primal, weighted_violation_for_dual

    def state_dict(self):
        return {
            "formulation_type": self.formulation_type,
            "augmented_lagrangian_scheduler_state_dict": self.augmented_lagrangian_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.formulation_type = state_dict["formulation_type"]
        self.augmented_lagrangian_scheduler.load_state_dict(state_dict["augmented_lagrangian_scheduler_state_dict"])

    def __repr__(self):
        return f"AugmentedLagrangianFormulation"


class FormulationType(Enum):
    PENALIZED = functools.partial(Formulation, formulation_type="penalized")
    LAGRANGIAN = functools.partial(Formulation, formulation_type="lagrangian")
    AUGMENTED_LAGRANGIAN = AugmentedLagrangianFormulation
