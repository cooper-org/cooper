# coding: utf8
"""
Implementation of the :py:class:`ExtrapolationConstrainedOptimizer` class.
"""

from typing import Callable, List, Optional, Union

import torch

from cooper.cmp import CMPState
from cooper.constraints import ConstraintGroup

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

    extrapolation = True
    alternating = False

    def __init__(
        self,
        constraint_groups: Union[List[ConstraintGroup], ConstraintGroup],
        primal_optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer],
        dual_optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer],
    ):

        super().__init__(constraint_groups, primal_optimizers, dual_optimizers)

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

        are_primal_extra_optims = [hasattr(_, "extrapolation") for _ in self.primal_optimizers]
        are_dual_extra_optims = [hasattr(_, "extrapolation") for _ in self.dual_optimizers]

        if not all(are_primal_extra_optims) or not all(are_dual_extra_optims):
            raise RuntimeError(
                """Some of the provided optimizers do not have an extrapolation method.
                Please ensure that all optimizers are extrapolation capable."""
            )

        are_augmented_lagrangian = [
            c.formulation.formulation_type == "augmented_lagrangian" for c in self.constraint_groups
        ]

        if any(are_augmented_lagrangian):
            raise NotImplementedError(
                """It is currently not supported to use extrapolation and an
                Augmented Lagrangian formulation. This feature is untested."""
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
            raise RuntimeError("Closure must be provided to step when using extrapolation.")

        # Store parameter copy and compute t+1/2 iterates
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.extrapolation()  # type: ignore

        # Call to dual_step flips sign of gradients, then triggers call to
        # dual_optimizer.extrapolation and applies post_step_
        self.dual_step(call_extrapolation=True)

        # Zero gradients and recompute loss at extrapolated point
        self.zero_grad()

        # The closure is re-evaluated here as the closure arguments may have changed
        # during the extrapolation step.
        extrapolated_cmp_state = closure(*closure_args, **closure_kwargs)
        _ = extrapolated_cmp_state.populate_lagrangian()

        # Populate gradients at extrapolation point
        extrapolated_cmp_state.backward()

        # After this, the calls to `step` will update the stored copies with
        # the newly computed gradients
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.step()

        self.dual_step(call_extrapolation=False)

    def dual_step(self, call_extrapolation=False):

        # TODO(juan43ramirez): except for call_extrapolation, this function is the same
        # as the one in SimultaneousConstrainedOptimizer.

        for constraint, dual_optimizer in zip(self.constraint_groups, self.dual_optimizers):

            # TODO(juan43ramirez): which convention will we use to access the gradient
            # of the multipliers?
            _multiplier = constraint.multiplier

            if _multiplier.weight.grad is not None:

                # Flip gradients for multipliers to perform ascent.
                # We only do the flipping *right before* applying the optimizer step to
                # avoid accidental double sign flips.
                _multiplier.weight.grad.mul_(-1.0)

                if call_extrapolation:
                    dual_optimizer.extrapolation()  # type: ignore
                else:
                    dual_optimizer.step()

                # Select the indices of multipliers corresponding to feasible constraints
                if _multiplier.implicit_constraint_type == "ineq":
                    # TODO(juan43ramirez): could alternatively access the state of the
                    # constraint, which would be more transparent.

                    # Feasibility is attained when the violation is negative. Given that
                    # the gradient sign is flipped, a negative violation corresponds to
                    # a positive gradient.
                    feasible_indices = _multiplier.weight.grad > 0.0
                else:
                    # TODO(juan43ramirez): add comment
                    feasible_indices = None

                _multiplier.post_step_(feasible_indices, restart_value=0.0)
