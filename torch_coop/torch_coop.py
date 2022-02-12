"""Main module."""

import logging
import warnings
from dataclasses import dataclass
from typing import List, Optional

import torch
from .multipliers import DenseMultiplier


@dataclass
class ClosureState:
    """Represents the 'value' of a given solution based on its loss and
    constraint violations.
    """

    loss: torch.Tensor
    eq_defect: Optional[List[torch.Tensor]] = None
    ineq_defect: Optional[List[torch.Tensor]] = None

    def as_tuple(self) -> tuple:
        return self.loss, self.eq_defect, self.ineq_defect


class OldConstrainedOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        primal_optimizer: torch.optim.Optimizer,
        dual_optimizer: Optional[torch.optim.Optimizer] = None,
        ineq_init: Optional[torch.Tensor] = None,
        eq_init: Optional[torch.Tensor] = None,
        aug_lag_coefficient=False,
        alternating=False,
        dual_restarts=False,
        verbose=False,
    ):

        # TODO: assert that if any optimizer has extrapolation, all do.
        self.primal_optimizer = primal_optimizer
        self.dual_optimizer = dual_optimizer

        # Convenience flag for determining if we actually deal with a constrained problem
        self.is_constrained = self.dual_optimizer is not None

        super().__init__(self.primal_optimizer.param_groups, {})

        # Flag to determine if dual variables have been initialized
        self.dual_init_done = False

        # Store user-provided initializations for dual variables
        self.ineq_init = ineq_init
        self.eq_init = eq_init

        if self.is_constrained:
            logging.info("Constrained Execution")

            # The dual optimizer is instantiated in 'initialize_dual_variables'
            self.dual_restarts = dual_restarts

            # Other optimization and Lagrangian options
            self.aug_lag_coefficient = aug_lag_coefficient
            self.alternating = alternating
        else:
            logging.info("Unconstrained Execution")

            self.dual_restarts = False

            # Override any given values for constrained optimization parameters
            self.aug_lag_coefficient = 0.0
            self.alternating = False

        self.verbose = verbose

    def step(self, closure):

        closure_state = closure()
        loss, eq_defect, ineq_defect = closure_state.as_tuple()

        # If not done before, instantiate and initialize dual variables
        # This step also instantiates dual_optimizer, if necessary
        if not self.dual_init_done and self.is_constrained:
            self.initialize_dual_variables(eq_defect, ineq_defect)

        # Compute Lagrangian based on current loss and values of multipliers
        lagrangian = self.lagrangian_backward(loss, eq_defect, ineq_defect)
        # TODO: do we still want to log the Lagrangian value?
        # closure_dict["lagrangian"] = lagrangian

        # TODO: Why was this being applied on the object loss?
        # Shouldn't this be called with input Lagrangian? Otherwise subsequent
        # extrapolation backprops will ignore constraints.
        self.run_optimizers_step(lagrangian, closure)

        return closure_state

    def run_optimizers_step(self, loss, closure_fn):

        should_back_prop = False
        if hasattr(self.primal_optimizer, "extrapolation"):
            self.primal_optimizer.extrapolation(loss)
            should_back_prop = True

        if (
            self.is_constrained
            and not self.alternating
            and hasattr(self.dual_optimizer, "extrapolation")
        ):
            self.dual_optimizer.extrapolation(loss)
            should_back_prop = True

        if should_back_prop:
            closure_state = closure_fn()
            lagrangian_ = self.lagrangian_backward(
                *closure_state.as_tuple(), ignore_primal=False
            )

        self.primal_optimizer.step()

        if self.is_constrained:

            if self.alternating:
                # Once having updated primal parameters, re-compute gradient
                # Skip gradient wrt model parameters to avoid wasteful computation
                # as we only need gradient wrt multipliers.
                closure_state = closure_fn()
                lagrangian_ = self.lagrangian_backward(
                    *closure_state.as_tuple(), ignore_primal=True
                )

            self.dual_step()

    def dual_step(self):
        # Update multipliers based on current constraint violations (gradients)
        self.dual_optimizer.step()

        if self.dual_restarts:
            # 'Reset' value of inequality multipliers to zero as soon as
            # solution becomes feasible
            if self.ineq_multipliers is not None:
                self.restart_dual_variables()

        # Apply projection step to inequality multipliers
        if self.ineq_multipliers is not None:
            self.ineq_multipliers.project_()

    def restart_dual_variables(self):
        # Call to lagrangian_backward has already flipped sign
        # Currently positive sign means original defect is negative => feasible

        feasible_filter = self.ineq_multipliers.weight.grad > 0

        self.ineq_multipliers.weight.grad[feasible_filter] = 0.0
        self.ineq_multipliers.weight.data[feasible_filter] = 0.0

    def lagrangian_backward(self, loss, eq_defect, ineq_defect, ignore_primal=False):
        """Compute Lagrangian and backward pass"""
        self.primal_optimizer.zero_grad()

        if self.is_constrained:
            self.dual_optimizer.zero_grad()

        lagrangian = self.compute_lagrangian(loss, eq_defect, ineq_defect)

        # Compute gradients
        if ignore_primal and self.is_constrained:
            # Only compute gradients wrt Lagrange multipliers
            lagrangian.backward(inputs=self.dual_params)
        else:
            lagrangian.backward()

        # Flip gradients for dual variables to perform ascent
        if self.is_constrained:
            [_.grad.mul_(-1.0) for _ in self.dual_params]

        return lagrangian.item()

    def compute_lagrangian(self, loss, eq_defect, ineq_defect):

        # Compute contribution of the constraint violations, weighted by the
        # current multiplier values
        weighted_violation = 0.0

        if eq_defect is not None:
            weighted_violation += torch.sum(self.eq_multipliers() * eq_defect)

        if ineq_defect is not None:
            weighted_violation += torch.sum(self.ineq_multipliers() * ineq_defect)

        # Lagrangian = loss + \sum_i multiplier_i * defect_i
        lagrangian = loss + weighted_violation

        # If using augmented Lagrangian, add squared sum of constraints
        # Following the formulation on Marc Toussaint slides (p 17-20)
        # https://ipvs.informatik.uni-stuttgart.de/mlr/marc/teaching/13-Optimization/03-constrainedOpt.pdf
        if self.aug_lag_coefficient > 0:

            if eq_defect is not None:
                eq_square = torch.sum(torch.square(eq_defect))
            else:
                eq_square = 0.0

            if ineq_defect is not None:
                ineq_filter = (ineq_defect >= 0) + (self.ineq_multipliers() > 0)
                ineq_square = torch.sum(torch.square(ineq_defect[ineq_filter]))
            else:
                ineq_square = 0.0

            lagrangian += self.aug_lag_coefficient * (eq_square + ineq_square)

        return lagrangian

    def initialize_dual_variables(
        self,
        eq_defect: Optional[torch.Tensor],
        ineq_defect: Optional[torch.Tensor],
    ):
        """Initialize dual variables and optimizers given list of equality and
        inequality defects.

        Args:
            eq_defect: Defects for equality constraints
            ineq_defect: Defects for inequality constraints.
        """

        if self.verbose:
            logging.info("Initializing dual variables")

        aux_dict = {
            "eq": {"defect": eq_defect, "init": self.eq_init},
            "ineq": {"defect": ineq_defect, "init": self.ineq_init},
        }

        dual_params = []
        for constraint_type in aux_dict.keys():
            defect = aux_dict[constraint_type]["defect"]
            init = aux_dict[constraint_type]["init"]

            mult_name = constraint_type + "_multipliers"

            if defect is None:
                self.state[mult_name] = None
            else:
                if init is None:
                    # If not provided custom initialization, Lagrange multipliers
                    # are initialized at 0

                    # This already preserves dtype and device of defect
                    casted_init = torch.zeros_like(defect)
                else:
                    casted_init = torch.tensor(
                        init,
                        device=defect.device,
                        dtype=defect.dtype,
                    )
                    assert defect.shape == casted_init.shape

                # Enforce positivity if dealing with inequality
                is_positive = constraint_type == "ineq"
                self.state[mult_name] = DenseMultiplier(
                    casted_init, positive=is_positive
                )

                dual_params.append(*self.state[mult_name].parameters())

        # Initialize dual optimizer in charge of newly created dual parameters
        self.dual_optimizer = self.dual_optimizer(self.dual_params)

        # Mark dual instantiation an init as complete
        self.dual_init_done = True

    def eval_multipliers(self, mult_type="ineq"):
        return self.state[mult_type + "_multipliers"].forward().data

    @property
    def ineq_multipliers(self):
        return self.state["ineq_multipliers"]

    @property
    def eq_multipliers(self):
        return self.state["eq_multipliers"]

    @property
    def dual_params(self):
        all_duals = []
        for _ in [self.eq_multipliers, self.ineq_multipliers]:
            if _ is not None:
                all_duals.append(_.weight)
        return all_duals


# def constraint_dot(defect, multiplier):
#     """Compute constraint contribution for given (potent. sparse) defect and multiplier"""
#     if False and defect.is_sparse:
#         hi = defect.coalesce()
#         indices = hi.indices().squeeze(0)
#         return torch.einsum(
#             "bh,bh->", multiplier(indices).to(dtype=hi.dtype), hi.values()
#         )
#     else:
#         return torch.sum(multiplier() * defect)
