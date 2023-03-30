import warnings
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch

FORMULATION_TYPE = Literal["penalized", "lagrangian", "augmented_lagrangian"]


class Formulation:
    def __init__(
        self,
        formulation_type: FORMULATION_TYPE,
        augmented_lagrangian_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        # TODO(gallego-posada): Add documentation

        if formulation_type not in ["penalized", "lagrangian", "augmented_lagrangian"]:
            raise ValueError(f"Formulation type {formulation_type} not understood.")

        if (formulation_type == "augmented_lagrangian") and (augmented_lagrangian_scheduler is None):
            raise ValueError("An augmented Lagrangian formulation requires an augmented Lagrangian scheduler.")

        if (augmented_lagrangian_scheduler is not None) and (formulation_type != "augmented_lagrangian"):
            raise ValueError(f"An augmented Lagrangian scheduler is not compatible with {formulation_type}.")

        self.formulation_type = formulation_type
        self.augmented_lagrangian_scheduler = augmented_lagrangian_scheduler

    def compute_lagrangian_contribution(
        self, constraint_type, multiplier, violation, strict_violation
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

            # TODO(gallego-posada): Augmented lagrangian is currently untested. Code
            # below may be unreliable.
            raise NotImplementedError("Augmented Lagrangian is currently untested.")

            if constraint_type == "ineq":
                # Compute filter based on strict constraint violation
                const_filter = torch.logical_or(strict_violation >= 0, multiplier > 0)
                sq_violation = torch.sum(const_filter.detach() * (violation**2))
            elif constraint_type == "eq":
                # Equality constraints do not need to be filtered
                sq_violation = torch.sum(violation**2)
            else:
                raise ValueError(f"{constraint_type} is incompatible with formulation_type=augmented_lagrangian")

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

    def state_dict(self):
        return {
            "formulation_type": self.formulation_type,
            "augmented_lagrangian_scheduler_state_dict": self.augmented_lagrangian_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.formulation_type = state_dict["formulation_type"]
        self.augmented_lagrangian_scheduler.load_state_dict(state_dict["augmented_lagrangian_scheduler_state_dict"])

    def __repr__(self):
        return f"Formulation(formulation_type={self.formulation_type}, has_scheduler={self.augmented_lagrangian_scheduler is not None})"
