# coding: utf8
"""
Implementation of the :py:class:`AlternatingConstrainedOptimizer` class.
"""

from typing import Callable, List, Optional, Union

import torch

from cooper.formulation import AugmentedLagrangianFormulation, Formulation
from cooper.problem import CMPState

from .constrained_optimizer import ConstrainedOptimizer


class AlternatingConstrainedOptimizer(ConstrainedOptimizer):
    def __init__(
        self,
        formulation: Formulation,
        primal_optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer],
        dual_optimizer: Optional[torch.optim.Optimizer] = None,
        dual_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        dual_restarts: bool = False,
    ):
        self.formulation = formulation
        self.cmp = self.formulation.cmp

        if isinstance(primal_optimizers, torch.optim.Optimizer):
            self.primal_optimizers = [primal_optimizers]
        else:
            self.primal_optimizers = primal_optimizers

        self.dual_optimizer = dual_optimizer
        self.dual_scheduler = dual_scheduler

        self.extrapolation = False
        self.alternating = True
        self.dual_restarts = dual_restarts

        self.base_sanity_checks()

    def step(
        self,
        closure: Optional[Callable[..., CMPState]] = None,
        *closure_args,
        defect_fn: Optional[Callable[..., CMPState]] = None,
        **closure_kwargs,
    ):
        """
        Performs a single optimization step on both the primal and dual
        variables.

        Args:
            closure: Closure ``Callable`` required for re-evaluating the
                objective and constraints when performing alternating updates.
                Defaults to None.

            *closure_args: Arguments to be passed to the closure function
                when re-evaluating.

            **closure_kwargs: Keyword arguments to be passed to the closure
                function when re-evaluating.
        """

        # If necessary, instantiate dual components
        if not hasattr(self.dual_optimizer, "param_groups"):
            self.instantiate_dual_optimizer_and_scheduler()

        assert not (
            (closure is None) and (defect_fn is None)
        ), "At least one of closure or defect_fn must be provided for alternating updates."

        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.step()

        self.populate_alternating_dual_gradient(
            closure, defect_fn, *closure_args, **closure_kwargs
        )

        self.dual_step()

    def populate_alternating_dual_gradient(
        self, closure, defect_fn, *closure_args, **closure_kwargs
    ):

        # Once having updated the primal parameters, re-compute gradient wrt
        # multipliers. Skip gradient wrt primal parameters to avoid wasteful
        # computation, as we only need gradient wrt multipliers.
        with torch.no_grad():

            assert closure is not None or defect_fn is not None

            if defect_fn is not None:
                alternate_cmp_state = defect_fn(*closure_args, **closure_kwargs)

                if alternate_cmp_state.loss is None:
                    # Store last computed loss
                    alternate_cmp_state.loss = self.formulation.cmp.state.loss

            elif closure is not None:
                alternate_cmp_state = closure(*closure_args, **closure_kwargs)

        # We have already computed the new CMP state with the new values of the
        # parameters. Now we only need to recalculate the Lagrangian so we can
        # get the gradients wrt the multipliers.
        # Note that the call to defect_fn might _not_ have populated the loss.
        # This is not a problem since we only need to compute the gradient wrt
        # the multipliers.
        if isinstance(self.formulation, AugmentedLagrangianFormulation):
            # Use LR of dual optimizer as penalty coefficient for the augmented
            # Lagrangian
            _ = self.formulation.composite_objective(
                closure=None,
                aug_lag_coeff_scheduler=self.dual_scheduler,
                pre_computed_state=alternate_cmp_state,
                write_state=True,
            )  # type: ignore
        else:
            _ = self.formulation.composite_objective(
                closure=None, pre_computed_state=alternate_cmp_state, write_state=True
            )  # type: ignore
        # Zero-out gradients for dual variables since they were already
        # populated earlier. We also zero-out primal gradients for safety
        # although not really necessary.
        self.zero_grad(ignore_primal=False, ignore_dual=False)

        # Not passing lagrangian since we only want to update the gradients for
        # the dual variables
        self.formulation._populate_gradients(
            lagrangian=None, ignore_primal=True, ignore_dual=False
        )

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
