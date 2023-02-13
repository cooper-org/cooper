import warnings
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch

from cooper import multipliers

CONSTRAINT_TYPE = Literal["eq", "ineq", "penalty"]
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
        self.formulation = self.build_formulation(formulation_type, formulation_kwargs)

        if multiplier is None:
            multiplier = self.build_multiplier(shape=shape, dtype=dtype, device=device, is_sparse=is_sparse)
        self.sanity_check_multiplier(multiplier)
        self.multiplier = multiplier

    def build_formulation(self, formulation_type, formulation_kwargs):
        if self.constraint_type == "penalty":
            # `penalty` constraints must be paired with "penalized" formulations. If no formulation is provided, we
            # default to a "penalized" formulation.
            if formulation_type != "penalized":
                warning_message = (
                    "A constraint of type `penalty` must be used with a `penalized` formulation, but received"
                    f" formulation_type={formulation_type}. The formulation type will be set to `penalized`."
                    " Please review your configuration and override the default formulation_type='lagrangian'."
                )
                warnings.warn(warning_message)
            formulation_type = "penalized"

        return Formulation(formulation_type, **formulation_kwargs)

    def sanity_check_multiplier(self, multiplier: multipliers.MULTIPLIER_TYPE) -> None:

        if (self.constraint_type == "penalty") != isinstance(multiplier, multipliers.ConstantMultiplier):
            # If a penalty "constraint" is used, then we must have been provided a ConstantMultiplier.
            raise ValueError("A ConstantMultiplier must be provided along with a `penalty` constraint.")

        if isinstance(multiplier, multipliers.ConstantMultiplier):
            if any(multiplier() < 0) and (self.constraint_type == "ineq"):
                raise ValueError("All entries of ConstantMultiplier must be non-negative for inequality constraints.")

        if isinstance(multiplier, multipliers.ExplicitMultiplier):
            if multiplier.implicit_constraint_type != self.constraint_type:
                raise ValueError(f"Provided multiplier is inconsistent with {self.constraint_type} constraint.")

    @property
    def state(self) -> ConstraintState:
        return self._state

    @state.setter
    def state(self, value: ConstraintState) -> None:
        if isinstance(self.multiplier, (multipliers.SparseMultiplier, multipliers.ImplicitMultiplier)):
            if value.constraint_features is None:
                raise ValueError(f"Multipliers of type {type(self.multiplier)} expect constraint features.")

        self._state = value

    def build_multiplier(self, shape: int, dtype: torch.dtype, device: torch.device, is_sparse: bool) -> None:

        multiplier_class = multipliers.SparseMultiplier if is_sparse else multipliers.DenseMultiplier

        tensor_factory = dict(dtype=dtype, device=device)
        tensor_factory["size"] = (shape, 1) if is_sparse else (shape,)

        return multiplier_class(init=torch.zeros(**tensor_factory), enforce_positive=(self.constraint_type == "ineq"))

    def compute_lagrangian_contribution(
        self, constraint_state: Optional[ConstraintState] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if constraint_state is None:
            constraint_state = self.state

        if constraint_state is None:
            raise ValueError("A `ConstraintState` (provided or internal) is needed to compute Lagrangian contribution")

        if constraint_state.constraint_features is None:
            multiplier_value = self.multiplier()
        else:
            multiplier_value = self.multiplier(constraint_state.constraint_features)

        # Strict violation represents the "actual" violation of the constraint.
        # We use it to update the value of the multiplier.
        if constraint_state.strict_violation is not None:
            strict_violation = constraint_state.strict_violation
        else:
            # If strict violation is not provided, we use the differentiable
            # violation (which always exists).
            strict_violation = constraint_state.violation

        primal_contribution, dual_contribution = self.formulation.compute_lagrangian_contribution(
            multiplier_value, constraint_state.violation, strict_violation
        )

        return multiplier_value, primal_contribution, dual_contribution

    def __repr__(self):
        return f"ConstraintGroup(constraint_type={self.constraint_type}, formulation={self.formulation}, multiplier={self.multiplier})"


class Formulation:
    def __init__(
        self,
        formulation_type: FORMULATION_TYPE,
        augmented_lagrangian_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):

        if formulation_type not in ["penalized", "lagrangian", "augmented_lagrangian"]:
            raise ValueError(f"Formulation type {formulation_type} not understood.")

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
