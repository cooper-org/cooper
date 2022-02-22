"""Lagrangian formulation"""

import abc
import logging
from typing import List, Optional

import torch

from .multipliers import DenseMultiplier
from .problem import Formulation


class BaseLagrangianFormulation(Formulation, metaclass=abc.ABCMeta):
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

        self.state_update: List[torch.Tensor] = []

        if aug_lag_coefficient < 0:
            raise ValueError("Augmented Lagrangian coefficient must be non-negative.")
        self.aug_lag_coefficient = aug_lag_coefficient

    def state(self):
        """Evaluates and returns all multipliers"""
        ineq_state = None if self.ineq_multipliers is None else self.ineq_multipliers()
        eq_state = None if self.eq_multipliers is None else self.eq_multipliers()

        return ineq_state, eq_state

    def no_none_state(self):
        """Returns self.state with Nones removed. Used for passing as inputs to
        backwards."""
        return [_ for _ in self.state() if _ is not None]

    @property
    def dual_parameters(self):
        all_duals = []

        if self.eq_multipliers is not None:
            all_duals += [self.eq_multipliers()]

        if self.ineq_multipliers is not None:
            all_duals += [self.ineq_multipliers()]

        return all_duals

    def create_state(self, cmp_state):
        """Initialize dual variables and optimizers given list of equality and
        inequality defects.

        Args:
            eq_defect: Defects for equality constraints
            ineq_defect: Defects for inequality constraints.
        """

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

    @abc.abstractmethod
    def get_composite_objective(self):
        pass

    @abc.abstractmethod
    def populate_gradients(self):
        pass

    def purge_state_update(self):
        self.state_update = []

    def weighted_violation(self, cmp_state, constraint_type):

        defect = getattr(cmp_state, constraint_type + "_defect")
        has_defect = defect is not None

        proxy_defect = getattr(cmp_state, "proxy_" + constraint_type + "_defect")
        has_proxy_defect = proxy_defect is not None

        if not has_proxy_defect:
            # If not given proxy constraints, then the regular defects are
            # used for computing gradients and evaluating the multipliers
            proxy_defect = defect

        if not has_defect:
            # We should always have at least the regular defects, if not, then
            # the problem instance does not have `constraint_type` constraints
            proxy_violation = 0.0
        else:
            multipliers = getattr(self, constraint_type + "_multipliers")()

            # We compute (primal) gradients of this object
            proxy_violation = torch.sum(multipliers.detach() * proxy_defect)

            # This is the violation of the "actual" constraint. We use this
            # to update the value of the multipliers by lazily filling the
            # multiplier gradients in `populate_gradients`
            violation_for_update = torch.sum(multipliers * defect.detach())
            self.state_update.append(violation_for_update)

        return proxy_violation


class LagrangianFormulation(BaseLagrangianFormulation):
    def get_composite_objective(self):

        cmp_state = self.cmp.state

        # Extract values from ProblemState object
        loss = cmp_state.loss
        ineq_defect, eq_defect = cmp_state.ineq_defect, cmp_state.eq_defect

        if self.cmp.is_constrained:
            # Compute contribution of the constraint violations, weighted by the
            # current multiplier values

            # If given proxy constraints, these are used to compute the terms
            # added to the Lagrangian, and the multiplier updates are based on
            # the non-proxy violations.
            # If not given proxy constraints, then gradients and multiplier updates
            # are based on the "regular" constraints.
            ineq_viol = self.weighted_violation(cmp_state, "ineq")
            eq_viol = self.weighted_violation(cmp_state, "eq")

            # Lagrangian = loss + \sum_i multiplier_i * defect_i
            lagrangian = loss + ineq_viol + eq_viol

            # TODO (1): verify that current implementation of proxy constraints
            # works properly with augmented lagrangian below.

            # If using augmented Lagrangian, add squared sum of constraints
            # Following the formulation on Marc Toussaint slides (p 17-20)
            # https://ipvs.informatik.uni-stuttgart.de/mlr/marc/teaching/13-Optimization/03-constrainedOpt.pdf
            if self.aug_lag_coefficient > 0:

                # TODO (2): I guess one would like to filter based on non-proxy
                # feasibility but then penalize based on the proxy constraint
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
            lagrangian = cmp_state.loss

        return lagrangian

    def populate_gradients(self, lagrangian, ignore_primal=False):
        # ignore_primal is used for alternating updates

        if ignore_primal and self.cmp.is_constrained:
            # Only compute gradients wrt Lagrange multipliers
            # No need to call backward on Lagrangian as the dual variables have
            # been detached when computing the `weighted_violation`s
            pass
        else:
            # Compute gradients wrt primal parameters only.
            # The gradient for the dual variables is computed based on the
            # non-proxy violations below.
            lagrangian.backward()

        # Fill in the gradients for the dual variables based on the violation of
        # the non-proxy constraint
        # This is equivalent to setting `dual_vars.grad = defect`
        if self.cmp.is_constrained:
            for violation_for_update in self.state_update:

                violation_for_update.backward(inputs=self.no_none_state())
