# coding: utf8
"""
Implementation of the :py:class:`ExtrapolationConstrainedOptimizer` class.
"""

from typing import Callable, List, Optional, Union

import torch

from cooper.formulation import AugmentedLagrangianFormulation, Formulation
from cooper.optim.extra_optimizers import ExtragradientOptimizer
from cooper.problem import CMPState

from .constrained_optimizer import ConstrainedOptimizer


class ExtrapolationConstrainedOptimizer(ConstrainedOptimizer):

    """
    Optimizes a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`
    given a provided :py:class:`~cooper.formulation.Formulation` using
    optimizers primal and dual optimizers capable of performing extrapolation.

    Args:
        formulation: ``Formulation`` of the ``ConstrainedMinimizationProblem``
            to be optimized.

        primal_optimizers: Fully instantiated
            :py:class:`~cooper.optim.extra_optimizers.ExtragradientOptimizer`\\s
            used to optimize the primal parameters (e.g. model parameters). The
            primal parameters can be partitioned into multiple optimizers, in
            which case ``primal_optimizers`` accepts a list of
            ``ExtragradientOptimizer``\\s.

        dual_optimizer: Partially instantiated
            :py:class:`~cooper.optim.extra_optimizers.ExtragradientOptimizer`
            used to optimize the dual variables (e.g. Lagrange multipliers).

        dual_scheduler: Partially instantiated
            ``torch.optim.lr_scheduler._LRScheduler`` used to schedule the
            learning rate of the dual variables. Defaults to None.

        dual_restarts: If True, perform "restarts" on the Lagrange
            multipliers associated with inequality constraints: whenever the
            constraint is satisfied, directly set the multiplier to zero.
            Defaults to False.

    """

    def __init__(
        self,
        formulation: Formulation,
        primal_optimizers: Union[List[ExtragradientOptimizer], ExtragradientOptimizer],
        dual_optimizer: ExtragradientOptimizer,
        dual_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        dual_restarts: bool = False,
    ):
        self.formulation = formulation

        if isinstance(primal_optimizers, ExtragradientOptimizer):
            self.primal_optimizers = [primal_optimizers]
        else:
            self.primal_optimizers = primal_optimizers

        self.dual_optimizer = dual_optimizer
        self.dual_scheduler = dual_scheduler

        self.dual_restarts = dual_restarts

        self.extrapolation = True
        self.alternating = False

        self.base_sanity_checks()

        self.custom_sanity_checks()

    def custom_sanity_checks(self):
        """
        Perform sanity checks on the initialization of ``ConstrainedOptimizer``.

        Raises:
            RuntimeError: Tried to construct an
                ExtrapolationConstrainedOptimizer but none of the primal
                optimizers has an extrapolation function.
            RuntimeError: One of the ``primal_optimizers`` has an
                ``extrapolation`` function while another does not. Please ensure
                that all primal optimizers agree on whether to perform
                extrapolation.
            RuntimeError: The ``primal_optimizers`` have extrapolation
                functions while the ``dual_optimizer`` does not. Extrapolation
                on only one player is not supported.
            NotImplementedError: It is currently not supported to use
                extrapolation and an augmented Lagrangian formulation. This
                feature is untested.
        """

        are_extra_optims = [hasattr(_, "extrapolation") for _ in self.primal_optimizers]

        if not any(are_extra_optims):
            raise RuntimeError(
                """Tried to construct an ExtrapolationConstrainedOptimizer but
                none of the primal optimizers has an extrapolation function."""
            )

        if any(are_extra_optims) and not all(are_extra_optims):
            raise RuntimeError(
                """One of the primal optimizers has an extrapolation function
                while another does not. Please ensure that all primal optimizers
                agree on whether to perform extrapolation."""
            )

        if not hasattr(self.dual_optimizer, "extrapolation"):
            raise RuntimeError(
                """Dual optimizer does not have an extrapolation function.
                Extrapolation on only one player is not supported."""
            )

        if isinstance(self.formulation, AugmentedLagrangianFormulation):
            raise NotImplementedError(
                """It is currently not supported to use extrapolation and an
                augmented Lagrangian formulation. This feature is untested."""
            )

    def step(
        self,
        closure: Optional[Callable[..., CMPState]] = None,
        *closure_args,
        defect_fn: Optional[Callable[..., CMPState]] = None,
        **closure_kwargs,
    ):
        """
        Performs a single optimization step on both the primal and dual
        variables. If ``dual_scheduler`` is provided, a scheduler step is
        performed on the learning rate of the ``dual_optimizer``.

        Args:
            closure: Closure ``Callable`` required for re-evaluating the
                objective and constraints when performing extrapolating updates.
                Defaults to None.

            *closure_args: Arguments to be passed to the closure function
                when re-evaluating.

            **closure_kwargs: Keyword arguments to be passed to the closure
                function when re-evaluating.
        """

        if closure is None:
            raise RuntimeError(
                "Closure must be provided to step when using extrapolation."
            )

        # If necessary, instantiate dual components
        if not hasattr(self.dual_optimizer, "param_groups"):
            self.instantiate_dual_optimizer_and_scheduler()

        # Store parameter copy and compute t+1/2 iterates
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.extrapolation()  # type: ignore

        # Call to dual_step flips sign of gradients, then triggers
        # call to dual_optimizer.extrapolation and projects dual
        # variables
        self.dual_step(call_extrapolation=True)

        # Zero gradients and recompute loss at t+1/2
        self.zero_grad()

        # For extrapolation, we need closure args here as the parameter
        # values will have changed in the update applied on the
        # extrapolation step
        lagrangian = self.formulation.composite_objective(
            closure, *closure_args, **closure_kwargs
        )  # type: ignore

        # Populate gradients at extrapolation point
        self.formulation.backward(lagrangian)

        # After this, the calls to `step` will update the stored copies with
        # the newly computed gradients
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.step()

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
