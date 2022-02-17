import abc
from typing import Optional

import torch
from .problem import ConstrainedMinimizationProblem, Formulation


class ConstrainedOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        formulation: Formulation,
        primal_optimizer: torch.optim.Optimizer,
        dual_optimizer: Optional[torch.optim.Optimizer] = None,
        alternating=False,
        dual_restarts=False,
        verbose=False,
    ):

        self.formulation = formulation
        self.cmp = self.formulation.cmp
        self.primal_optimizer = primal_optimizer
        self.dual_optimizer = dual_optimizer

        self.alternating = alternating
        self.dual_restarts = dual_restarts

        super().__init__(self.primal_optimizer.param_groups, {})

    def custom_backward(self, lagrangian):
        self.formulation.populate_gradients(lagrangian)

    def composite_objective(self, closure):

        self.cmp.state = closure()

        # If not done before, instantiate and initialize dual variables
        if self.cmp.is_constrained and (not self.formulation.is_state_created):
            self.formulation.create_state(self.cmp.state)
            # Instantiates dual_optimizer
            self.dual_optimizer = self.dual_optimizer(self.formulation.dual_parameters)

        # Compute Lagrangian based on current loss and values of multipliers
        self.formulation.purge_state_update()
        lagrangian = self.formulation.get_composite_objective()

        return lagrangian

    def step(self, closure, previous_lagrangian=None):

        # TODO: integrate extrapolation
        # should_back_prop = False
        # if hasattr(self.primal_optimizer, "extrapolation"):
        #     self.primal_optimizer.extrapolation(previous_lagrangian)
        #     should_back_prop = True

        # if (
        #     self.is_constrained
        #     and not self.alternating
        #     and hasattr(self.dual_optimizer, "extrapolation")
        # ):
        #     self.dual_optimizer.extrapolation(previous_lagrangian)
        #     should_back_prop = True

        # if should_back_prop:
        #     closure_state = closure_fn()
        #     lagrangian_ = self.lagrangian_backward(
        #         *closure_state.as_tuple(), ignore_primal=False
        #     )

        self.primal_optimizer.step()

        if self.cmp.is_constrained:

            if self.alternating:
                # TODO: add test for this

                # Once having updated primal parameters, re-compute gradient
                # Skip gradient wrt model parameters to avoid wasteful computation
                # as we only need gradient wrt multipliers.
                with torch.no_grad():
                    self.cmp.state = closure()
                lagrangian = self.formulation.get_composite_objective(self.cmp)

                # Zero-out gradients for dual variables since they were already
                # populated earlier.
                # Also zeroing primal gradients for safety although not really
                # necessary.
                self.zero_grad(ignore_primal=False, ignore_dual=False)

                # Not passing lagrangian since we only want to update the
                # gradients for the dual variables
                self.formulation.populate_gradients(lagrangian=None, ignore_primal=True)

            self.dual_step()

    def dual_step(self):

        # Flip gradients for multipliers to perform ascent.
        # We only do the flipping *right before* applying the optimizer step to
        # avoid accidental double sign flips.
        for multiplier in self.formulation.state():
            if multiplier is not None:
                multiplier.grad.mul_(-1.0)

        # Update multipliers based on current constraint violations (gradients)
        self.dual_optimizer.step()

        if self.formulation.ineq_multipliers is not None:
            if self.dual_restarts:
                # "Reset" value of inequality multipliers to zero as soon as
                # solution becomes feasible
                self.restart_dual_variables()

            # Apply projection step to inequality multipliers
            self.formulation.ineq_multipliers.project_()

    def restart_dual_variables(self):
        # Call to formulation.populate_gradients has already flipped sign
        # A currently *positive* gradient means original defect is negative, so
        # the constraint is being satisfied.

        # The code below still works in the case of proxy constraints, since the
        # multiplier updates are computed based on *non-proxy* constraints
        feasible_filter = self.formulation.ineq_multipliers.weight.grad > 0
        self.formulation.ineq_multipliers.weight.grad[feasible_filter] = 0.0
        self.formulation.ineq_multipliers.weight.data[feasible_filter] = 0.0

    def zero_grad(self, ignore_primal=False, ignore_dual=False):

        if not ignore_primal:
            self.primal_optimizer.zero_grad()

        if not ignore_dual:

            if self.formulation.is_state_created:
                if self.dual_optimizer is None:
                    raise RuntimeError(
                        "Requested zeroing gradients but dual_optimizer is None."
                    )
                else:
                    self.dual_optimizer.zero_grad()
