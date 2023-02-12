from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch

from cooper import multipliers

CONSTRAINT_TYPE = Literal["eq", "ineq"]
FORMULATION_TYPE = Literal["penalized", "lagrangian", "augmented_lagrangian"]


@dataclass
class ConstraintState:
    """Constraint state."""

    violation: torch.Tensor
    strict_violation: Optional[torch.Tensor] = None
    constraint_features: Optional[torch.Tensor] = None


class ConstraintGroup:
    """Constraint Group."""

    def __init__(
        self,
        constraint_type: CONSTRAINT_TYPE,
        formulation_type: Optional[FORMULATION_TYPE] = "lagrangian",
        formulation_kwargs: Optional[dict] = {},
        multiplier: Optional[multipliers.MULTIPLIER_TYPE] = None,
        shape: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        is_sparse: Optional[bool] = False,
    ):

        self._state: ConstraintState = None

        self.constraint_type = constraint_type
        self.formulation = Formulation(formulation_type, **formulation_kwargs)

        if (formulation_type == "penalized") != isinstance(multiplier, multipliers.ConstantMultiplier):
            # This check implicitly verifies that if the formulation is penalized, then
            # we are receiving an instantiated ConstantMultiplier.
            raise ValueError("A ConstantMultiplier must be provided along with a `penalized` formulation.")

        if multiplier is not None:
            if isinstance(multiplier, multipliers.ExplicitMultiplier):
                multiplier_type = "ineq" if multiplier.positive else "eq"
                if multiplier_type != constraint_type:
                    raise ValueError(f"{multiplier_type} multiplier inconsistent with {constraint_type} constraint.")

            self.multiplier = multiplier
        else:
            self.initialize_multiplier(shape=shape, dtype=dtype, device=device, is_sparse=is_sparse)

    @property
    def state(self) -> ConstraintState:
        return self._state

    @state.setter
    def state(self, value: ConstraintState) -> None:
        if isinstance(self.multiplier, (multipliers.SparseMultiplier, multipliers.ImplicitMultiplier)):
            if value.constraint_features is None:
                raise ValueError(f"Multipliers of type {type(self.multiplier)} expect constraint features.")

        self._state = value

    def initialize_multiplier(self, shape: int, dtype: torch.dtype, device: torch.device, is_sparse: bool) -> None:

        tensor_factory = dict(dtype=dtype, device=device)
        tensor_factory["size"] = (shape, 1) if is_sparse else (shape,)

        positive = self.constraint_type == "ineq"
        multiplier_class = multipliers.SparseMultiplier if is_sparse else multipliers.DenseMultiplier
        multiplier_init = torch.zeros(**tensor_factory)

        self.multiplier = multiplier_class(init=multiplier_init, positive=positive)

    def evaluate_multiplier(self, *args) -> torch.Tensor:
        if isinstance(self.multiplier, (multipliers.DenseMultiplier, multipliers.ConstantMultiplier)):
            return self.multiplier.weight

        return self.multiplier(*args)

    def compute_lagrangian_contribution(
        self, constraint_state: Optional[ConstraintState] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if constraint_state is None:
            constraint_state = self.state

        if constraint_state is None:
            raise ValueError("Can not compute Lagrangian contribution with `None` constraint state.")

        multiplier_inputs = ()
        if constraint_state.constraint_features is not None:
            multiplier_inputs = (constraint_state.constraint_features,)

        multiplier = self.evaluate_multiplier(*multiplier_inputs)

        # Strict violation represents the "actual" violation of the constraint.
        # We use it to update the value of the multiplier.
        if constraint_state.strict_violation is not None:
            strict_violation = constraint_state.strict_violation
        else:
            # If strict violation is not provided, we use the differentiable
            # violation (which always exists).
            strict_violation = constraint_state.violation

        primal_contribution, dual_contribution = self.formulation.compute_lagrangian_contribution(
            multiplier, constraint_state.violation, strict_violation
        )

        return multiplier, primal_contribution, dual_contribution

    def __repr__(self):
        return f"ConstraintGroup(constraint_type={self.constraint_type}, formulation={self.formulation}, multiplier={self.multiplier})"


class Formulation:
    def __init__(
        self,
        formulation_type: FORMULATION_TYPE,
        augmented_lagrangian_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):

        if (formulation_type == "augmented_lagrangian") and (augmented_lagrangian_scheduler is None):
            raise ValueError("An augmented Lagrangian formulation requires an augmented Lagrangian scheduler.")

        if (augmented_lagrangian_scheduler is not None) and (formulation_type != "augmented_lagrangian"):
            raise ValueError(f"An augmented Lagrangian scheduler is not compatible with {formulation_type}.")

        self.formulation_type = formulation_type
        self.augmented_lagrangian_scheduler = augmented_lagrangian_scheduler

    def compute_lagrangian_contribution(
        self, multiplier, violation, strict_violation
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # When computing the gradient of the Lagrangian with respect to the
        # primal variables, we do not need to differentiate the multiplier.
        weighted_violation_for_primal = torch.sum(multiplier.detach() * violation)

        if self.formulation_type == "penalized":
            # Penalized formulations have no _trainable_ dual variables, so we adopt
            # the convention of setting this variable to None.
            weighted_violation_for_dual = None
        else:
            # When computing the gradient of the Lagrangian with respect to the dual
            # variables, we only need the _value_ of the constraint violation and
            # not its gradient. So we detach the violation to avoid computing its
            # gradient. Note that this enables the use of non-differentiable
            # constraints for updating the multiplier.
            #
            # This insight was originally proposed by Cotter et al. in the paper
            # "Optimization with Non-Differentiable Constraints with Applications to
            # Fairness, Recall, Churn, and Other Goals" under the name of "proxy"
            # constraints. (https://jmlr.org/papers/v20/18-616.html, Sec. 4.2)
            weighted_violation_for_dual = torch.sum(multiplier * strict_violation.detach())

        if self.formulation_type == "augmented_lagrangian":

            if self.constraint_type == "ineq":
                # Compute filter based on strict constraint violation
                const_filter = torch.logical_or(strict_violation >= 0, multiplier > 0)
                sq_violation = torch.sum(const_filter.detach() * (violation**2))
            else:
                # Equality constraints do not need to be filtered
                sq_violation = torch.sum(violation**2)

            # TODO(gallego-posada): Why were we doing this check before?
            # # Gather all the learning rates for the "parameter groups" of the dual
            # # variables, and check that all the learning rates are the same.
            dual_lrs = self.augmented_lagrangian_scheduler.get_last_lr()
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
                weighted_violation_for_primal += 0.5 * augmented_lagrangian_coefficient * sq_violation

        return weighted_violation_for_primal, weighted_violation_for_dual

    def __repr__(self):
        return f"Formulation(formulation_type={self.formulation_type}, has_scheduler={self.augmented_lagrangian_scheduler is not None})"
