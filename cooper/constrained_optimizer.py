from typing import Optional

import torch

from .problem import Formulation


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

        self.sanity_checks()

    def sanity_checks(self):

        is_alternating = self.alternating
        is_aug_lag = hasattr(self.formulation, "aug_lag_coefficient") and (
            self.formulation.aug_lag_coefficient > 0
        )

        # We assume that both optimizers agree on whether to use extrapolation
        # or not, so we use the primal optimizer as reference for deciding if
        # we use extrapolation. See check below for matching extrapolation behavior.
        self.is_extrapolation = hasattr(self.primal_optimizer, "extrapolation")

        if is_aug_lag and self.is_extrapolation:
            raise NotImplementedError(
                """It is currently not possible to use extrapolation and an
                augmented Lagrangian formulation"""
            )

        if is_alternating and self.is_extrapolation:
            raise RuntimeError(
                """Should not use extrapolation and alternating updates
                simultaneously. Please disable one of these two modes."""
            )

        if not (self.cmp.is_constrained) and (self.dual_optimizer is not None):
            raise RuntimeError(
                """Provided a dual optimizer, but the `Problem` class claims to
                be unconstrained."""
            )

        if not (self.cmp.is_constrained) and self.is_extrapolation:
            raise RuntimeError(
                """Using an extrapolating optimizer an unconstrained problem
                might result in unexpected behavior. Consider using a
                non-extrapolating optimizer instead."""
            )

        if hasattr(self.primal_optimizer, "extrapolation") != hasattr(
            self.dual_optimizer, "extrapolation"
        ):
            raise RuntimeError(
                """Primal and dual optimizers do not agree on whether to use
                extrapolation or not."""
            )

    def step(self, closure=None, *closure_args, **closure_kwargs):

        if self.cmp.is_constrained and not hasattr(self.dual_optimizer, "param_groups"):
            # Checks if needed and instantiates dual_optimizer
            self.dual_optimizer = self.dual_optimizer(self.formulation.dual_parameters)

        if self.is_extrapolation or self.alternating:
            assert closure is not None

        if self.is_extrapolation:
            # Store parameter copy and compute t+1/2 iterates
            self.primal_optimizer.extrapolation()
            if self.cmp.is_constrained:
                # Call to dual_step flips sign of gradients, then triggers call
                # to dual_optimizer.extrapolation and projects dual variables
                self.dual_step(call_extrapolation=True)

            # Zero gradients and recompute loss at t+1/2
            self.zero_grad()

            # For extrapolation, we need the closure args here as the parameter
            # values will have changed in the update applied on the extrapolation step
            lagrangian = self.formulation.composite_objective(
                closure, *closure_args, **closure_kwargs
            )

            # Populate gradients at extrapolation point
            self.formulation.custom_backward(lagrangian)

            # After this, the calls to `step` will update the stored copies with
            # the newly computed gradients
            self.primal_optimizer.step()
            if self.cmp.is_constrained:
                self.dual_step()

        else:

            self.primal_optimizer.step()

            if self.cmp.is_constrained:

                if self.alternating:
                    # TODO: add test for this

                    # Once having updated primal parameters, re-compute gradient
                    # Skip gradient wrt model parameters to avoid wasteful computation
                    # as we only need gradient wrt multipliers.
                    with torch.no_grad():
                        self.cmp.state = closure(*closure_args, **closure_kwargs)
                    lagrangian = self.formulation.get_composite_objective(self.cmp)

                    # Zero-out gradients for dual variables since they were already
                    # populated earlier.
                    # Also zeroing primal gradients for safety although not really
                    # necessary.
                    self.zero_grad(ignore_primal=False, ignore_dual=False)

                    # Not passing lagrangian since we only want to update the
                    # gradients for the dual variables
                    self.formulation.populate_gradients(
                        lagrangian=None, ignore_primal=True
                    )

                self.dual_step()

    def dual_step(self, call_extrapolation=False):

        # Flip gradients for multipliers to perform ascent.
        # We only do the flipping *right before* applying the optimizer step to
        # avoid accidental double sign flips.
        for multiplier in self.formulation.state():
            if multiplier is not None:
                multiplier.grad.mul_(-1.0)

        # Update multipliers based on current constraint violations (gradients)
        if call_extrapolation:
            self.dual_optimizer.extrapolation()
        else:
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
