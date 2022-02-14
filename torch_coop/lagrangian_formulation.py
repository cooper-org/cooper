"""Lagrangian formulation"""

import abc
from dataclasses import dataclass
import logging
from typing import List, Optional

import torch
from .problem import Formulation
from .multipliers import DenseMultiplier


class LagrangianFormulation(Formulation):
    def __init__(
        self,
        cmp,
        ineq_init: Optional[torch.Tensor] = None,
        eq_init: Optional[torch.Tensor] = None,
        aug_lag_coefficient: float = 0.0,
    ):
        """Construct new `LagrangianFormulation`"""

        self.cmp = cmp

        self.ineq_multipliers = None
        self.eq_multipliers = None

        # Store user-provided initializations for dual variables
        self.ineq_init = ineq_init
        self.eq_init = eq_init

        if aug_lag_coefficient < 0:
            raise ValueError("Augmented Lagrangian coefficient must be non-negative.")
        self.aug_lag_coefficient = aug_lag_coefficient

    def state(self):
        """Evaluates and returns all multipliers"""
        ineq_state = None if self.ineq_multipliers is None else self.ineq_multipliers()
        eq_state = None if self.eq_multipliers is None else self.eq_multipliers()

        return ineq_state, eq_state

    @property
    def dual_parameters(self):
        all_duals = []

        if self.eq_multipliers is not None:
            all_duals += [self.eq_multipliers()]

        if self.ineq_multipliers is not None:
            all_duals += [self.ineq_multipliers()]

        return all_duals

    def create_state(
        self,
        cmp_state
        # eq_defect: Optional[torch.Tensor] = None,
        # ineq_defect: Optional[torch.Tensor] = None,
    ):
        """Initialize dual variables and optimizers given list of equality and
        inequality defects.

        Args:
            eq_defect: Defects for equality constraints
            ineq_defect: Defects for inequality constraints.
        """

        # aux_dict = {}

        # Ensure that dual variables are not re-initialized
        for constraint_type in ["eq", "ineq"]:

            mult_name = constraint_type + "_multipliers"

            defect = getattr(cmp_state, constraint_type + "_defect")
            proxy_defect = getattr(cmp_state, "proxy_" + constraint_type + "_defect")

            has_defect = defect is not None
            has_proxy_defect = proxy_defect is not None

            if has_defect or has_proxy_defect:

                # Ensure dual variables have not been initialized previously
                assert getattr(self, constraint_type + "_multipliers") is None

                # If given proxy and non-proxy constraints, sanity-check tensor sizes
                if has_defect and has_proxy_defect:
                    assert defect.shape == proxy_defect.shape

                # Choose a tensor for getting device and dtype information
                defect_for_init = defect if has_defect else proxy_defect

                init_tensor = getattr(self, constraint_type + "_init")
                if init_tensor is None:
                    # If not provided custom initialization, Lagrange multipliers
                    # are initialized at 0

                    # This already preserves dtype and device of defect
                    casted_init = torch.zeros_like(defect_for_init)
                else:
                    casted_init = torch.tensor(
                        init_tensor,
                        device=defect_for_init.device,
                        dtype=defect_for_init.dtype,
                    )
                    assert defect_for_init.shape == casted_init.shape

                # Enforce positivity if dealing with inequality
                is_positive = constraint_type == "ineq"
                multiplier = DenseMultiplier(casted_init, positive=is_positive)

                setattr(self, mult_name, multiplier)

    @property
    def is_state_created(self):
        """Returns True if any Lagrange multipliers have been initialized"""
        return self.ineq_multipliers is not None or self.eq_multipliers is not None

    def get_composite_objective(self):

        problem_state = self.cmp.state

        # Extract values from ProblemState object
        loss = problem_state.loss
        ineq_defect, eq_defect = problem_state.ineq_defect, problem_state.eq_defect

        if self.cmp.is_constrained:
            # Compute contribution of the constraint violations, weighted by the
            # current multiplier values
            weighted_violation = 0.0

            if ineq_defect is not None:
                weighted_violation += torch.sum(self.ineq_multipliers() * ineq_defect)

            if eq_defect is not None:
                weighted_violation += torch.sum(self.eq_multipliers() * eq_defect)

            # Lagrangian = loss + \sum_i multiplier_i * defect_i
            lagrangian = loss + weighted_violation

            # If using augmented Lagrangian, add squared sum of constraints
            # Following the formulation on Marc Toussaint slides (p 17-20)
            # https://ipvs.informatik.uni-stuttgart.de/mlr/marc/teaching/13-Optimization/03-constrainedOpt.pdf
            if self.aug_lag_coefficient > 0:

                if ineq_defect is not None:
                    ineq_filter = (ineq_defect >= 0) + (self.ineq_multipliers() > 0)
                    ineq_square = torch.sum(torch.square(ineq_defect[ineq_filter]))
                else:
                    ineq_square = 0.0

                if eq_defect is not None:
                    eq_square = torch.sum(torch.square(eq_defect))
                else:
                    eq_square = 0.0

                lagrangian += self.aug_lag_coefficient * (ineq_square + eq_square)

        else:
            lagrangian = problem_state.loss

        return lagrangian

    def populate_gradients(self, lagrangian, ignore_primal=False):
        # ignore_primal is used for alternating updates

        # Compute gradients
        if ignore_primal and self.cmp.is_constrained:
            # Only compute gradients wrt Lagrange multipliers
            lagrangian.backward(inputs=self.state())
        else:
            lagrangian.backward()

        # Flip gradients for multipliers to perform ascent
        if self.cmp.is_constrained:
            for multiplier in self.state():
                if multiplier is not None:
                    multiplier.grad.mul_(-1.0)
