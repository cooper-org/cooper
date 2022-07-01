# coding: utf8
"""
Implementation of :py:class:`ConstrainedOptimizer` class, which has 2 main
methods:

- :py:meth:`~ConstrainedOptimizer.zero_grad`

- :py:meth:`~ConstrainedOptimizer.step`
"""

import pdb
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Type

import torch

from .problem import CMPState, Formulation
from .utils import validate_state_dicts


@dataclass
class ConstrainedOptimizerState:
    """Represents the "state" of a Constrained Optimizer in terms of the state
    dicts of the primal optimizer, as well as those of the dual optimizer and
    the dual scheduler if applicable. This is used for checkpointing.

    # TODO(JGP): Add docs about difference between this and FormulationState
    # here we focus on the optimizers, while the formulation contains dual vars.

    Args:
        primal_optimizer_state: State dict for the primal optimizer.
        dual_optimizer_state: State dict for the dual optimizer.
        dual_scheduler_state: State dict for the primal optimizer.
    """

    primal_optimizer_state: Dict
    dual_optimizer_state: Optional[Dict] = None
    dual_scheduler_state: Optional[Dict] = None
    alternating: bool = False
    dual_restarts: bool = False

    def __eq__(self, other):

        assert isinstance(other, ConstrainedOptimizerState)

        def compare_state_dicts(dict_name):
            try:
                return validate_state_dicts(
                    getattr(self, dict_name), getattr(other, dict_name)
                )
            except:
                return False

        state_dict_names = [
            "primal_optimizer_state",
            "dual_optimizer_state",
            "dual_scheduler_state",
        ]

        all_checks = [compare_state_dicts(_) for _ in state_dict_names]
        all_checks.append(self.alternating == other.alternating)
        all_checks.append(self.dual_restarts == other.dual_restarts)

        return all(all_checks)


class ConstrainedOptimizer:
    """
    Optimizes a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`
    given its :py:class:`~cooper.problem.Formulation`.

    A ``ConstrainedOptimizer`` includes one or two
    :class:`torch.optim.Optimizer`\\s, for the primal and dual variables
    associated with the ``Formulation``, respectively.

    A ``ConstrainedOptimizer`` can be used on constrained or unconstrained
    ``ConstrainedMinimizationProblem``\\s. Please refer to the documentation
    of the :py:class:`~cooper.problem.ConstrainedMinimizationProblem` and
    :py:class:`~cooper.problem.Formulation` classes for further details on
    handling unconstrained problems.

    Args:
        formulation: ``Formulation`` of the ``ConstrainedMinimizationProblem``
            to be optimized.

        primal_optimizer: Fully instantiated ``torch.optim.Optimizer`` used
            to optimize the primal parameters (e.g. model parameters).

        dual_optimizer: Partially instantiated ``torch.optim.Optimizer``
            used to optimize the dual variables (e.g. Lagrange multipliers).
            Defaults to None.
            When dealing with an unconstrained problem, should be set to None.

        dual_scheduler: Partially instantiated
            ``torch.optim.lr_scheduler._LRScheduler``
            used to schedule the learning rate of the dual variables.
            Defaults to None.
            When dealing with an unconstrained problem, should be set to None.

        alternating: Whether to alternate parameter updates between primal and
            dual parameters. Otherwise, do simultaneous parameter updates.
            Defaults to False.

        dual_restarts: If True, perform "restarts" on the Lagrange
            multipliers associated with inequality constraints: whenever the
            constraint is satisfied, directly set the multiplier to zero.
            Defaults to False.

    """

    def __init__(
        self,
        formulation: Formulation,
        primal_optimizer: torch.optim.Optimizer,
        dual_optimizer: Optional[torch.optim.Optimizer] = None,
        dual_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        alternating: bool = False,
        dual_restarts: bool = False,
    ):
        self.formulation = formulation
        self.cmp = self.formulation.cmp
        self.primal_optimizer = primal_optimizer
        self.dual_optimizer = dual_optimizer
        self.dual_scheduler = dual_scheduler

        self.alternating = alternating
        self.dual_restarts = dual_restarts

        self.sanity_checks()

    def sanity_checks(self):
        """
        Perform sanity checks on the initialization of ``ConstrainedOptimizer``.

        Raises:
            NotImplementedError: The ``Formulation`` has an augmented Lagrangian
                coefficient and ``primal_optimizer`` has an ``extrapolation``
                function. This is not supported because of possible unexpected
                behavior.
            RuntimeError: The ``primal_optimizer`` has an ``extrapolation``
                function and ``alternating`` was set to True. Mixing
                extrapolation and alternating updates is not supported.
            RuntimeError: a ``dual_optimizer`` was provided but the
                ``ConstrainedMinimizationProblem`` of formulation was
                unconstrained. There are no dual variables to optimize.
            RuntimeError: a ``dual_scheduler`` was provided but the
                ``ConstrainedMinimizationProblem`` of formulation was
                unconstrained. There are no dual variables and no
                ``dual_optimizer`` for learning rate scheduling.
            RuntimeError: a ``dual_scheduler`` was provided but no
                ``dual_optimizer`` was provided. Can not schedule the learning
                rate of an unknown optimizer.
            RuntimeError: the considered ``ConstrainedMinimizationProblem`` is
                unconstrained, but the provided ``primal_optimizer`` has an
                ``extrapolation`` function. This is not supported because of
                unexpected behavior when using extrapolation to update the
                primal parameters without any dual parameters.
            RuntimeError: One of ``primal_optimizer`` or ``dual_optimizer`` has
                an extrapolation function while the other does not.
                Extrapolation on only one player is not supported.
        """

        is_alternating = self.alternating
        is_aug_lag = hasattr(self.formulation, "aug_lag_coefficient") and (
            self.formulation.aug_lag_coefficient > 0
        )

        # We assume that both optimizers agree on whether to use extrapolation
        # or not, so we use the primal optimizer as reference for deciding
        # whether to use extrapolation. See check below for matching
        # extrapolation behavior.
        self.is_extrapolation = hasattr(self.primal_optimizer, "extrapolation")

        if is_alternating and self.dual_restarts:
            warnings.warn(
                """Using alternating updates with dual restarts is untested.
                Please use with caution."""
            )

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

        if self.dual_scheduler is not None:
            if not (self.cmp.is_constrained):
                raise RuntimeError(
                    """A dual scheduler was provided, but the `Problem` class
                    claims to be unconstrained."""
                )

            if self.dual_optimizer is None:
                raise RuntimeError(
                    """A dual scheduler was provided, but no dual optimizer
                    was provided."""
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
                objective and constraints when performing alternating or
                extrapolating updates.
                Defaults to None.

            *closure_args: Arguments to be passed to the closure function
                when re-evaluating.

            **closure_kwargs: Keyword arguments to be passed to the closure
                function when re-evaluating.
        """

        # TODO (JGP): The logic inside this method is becoming overly complex
        # due to the constant friction between extrapolation, alternating
        # updates, and proxy-constraints. We might want to consider refactoring.

        if self.cmp.is_constrained and not hasattr(self.dual_optimizer, "param_groups"):
            assert self.dual_optimizer is not None and callable(self.dual_optimizer)
            # Checks if needed and instantiates dual_optimizer
            self.dual_optimizer = self.dual_optimizer(self.formulation.dual_parameters)

            if self.dual_scheduler is not None:
                assert callable(self.dual_scheduler), "dual_scheduler must be callable"
                # Instantiates the dual_scheduler
                self.dual_scheduler = self.dual_scheduler(self.dual_optimizer)

        assert not (
            self.is_extrapolation and (closure is None)
        ), "Closure must be provided to step when using extrapolation"

        assert not (
            self.alternating and (closure is None) and (defect_fn is None)
        ), "At least one of closure or defect_fn must be provided for alternating updates"

        if self.is_extrapolation:
            # Store parameter copy and compute t+1/2 iterates
            self.primal_optimizer.extrapolation()  # type: ignore
            if self.cmp.is_constrained:
                # Call to dual_step flips sign of gradients, then triggers call
                # to dual_optimizer.extrapolation and projects dual variables
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
            self.formulation.custom_backward(lagrangian)

            # After this, the calls to `step` will update the stored copies with
            # the newly computed gradients
            self.primal_optimizer.step()
            if self.cmp.is_constrained:
                self.dual_step()

                if self.dual_scheduler is not None:
                    # Do a step on the dual scheduler after the actual step on
                    # the dual parameters. Intermediate updates that take
                    # place inside the extrapolation process do not perform a
                    # call to the scheduler's step method
                    self.dual_scheduler.step()

        else:
            # Non-extrapolation case

            self.primal_optimizer.step()

            if self.cmp.is_constrained:

                if self.alternating:
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
        # Call to formulation._populate_gradients has already flipped sign
        # A currently *positive* gradient means original defect is negative, so
        # the constraint is being satisfied.

        # The code below still works in the case of proxy constraints, since the
        # multiplier updates are computed based on *non-proxy* constraints
        feasible_filter = self.formulation.ineq_multipliers.weight.grad > 0
        self.formulation.ineq_multipliers.weight.grad[feasible_filter] = 0.0
        self.formulation.ineq_multipliers.weight.data[feasible_filter] = 0.0

    def zero_grad(self, ignore_primal: bool = False, ignore_dual: bool = False):
        """
        Sets the gradients of all optimized
        :py:class:`~torch.nn.parameter.Parameter`\\s to zero. This includes both
        the primal and dual variables.

        Args:
            ignore_primal: If True, the gradients of the primal variables will
                not be zeroed. Defaults to False.

            ignore_dual: If True, the gradients of the dual variables will not
                be zeroed. Defaults to False.
        """

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

    def state_dict(self) -> ConstrainedOptimizerState:
        """
        Returns the state of the ConstrainedOptimizer. See
        :py:class:`~cooper.constrained_optimizer.ConstrainedOptimizerState`.
        """

        primal_optimizer_state = self.primal_optimizer.state_dict()

        if self.dual_optimizer is not None:
            dual_optimizer_state = self.dual_optimizer.state_dict()
        else:
            dual_optimizer_state = None

        if self.dual_scheduler is not None:
            dual_scheduler_state = self.dual_scheduler.state_dict()
        else:
            dual_scheduler_state = None

        return ConstrainedOptimizerState(
            primal_optimizer_state=primal_optimizer_state,
            dual_optimizer_state=dual_optimizer_state,
            dual_scheduler_state=dual_scheduler_state,
            alternating=self.alternating,
            dual_restarts=self.dual_restarts,
        )

    @classmethod
    def load_from_state_dict(
        cls,
        const_optim_state: ConstrainedOptimizerState,
        formulation: Formulation,
        primal_optimizer: torch.optim.Optimizer,
        dual_optimizer_class: Type[torch.optim.Optimizer] = None,
        dual_scheduler_class: Type[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """
        Loads the state of the ConstrainedOptimizer. This method should be called

        Args:
            state_dict: state of the ConstrainedOptimizer.
        """

        primal_optimizer.load_state_dict(const_optim_state.primal_optimizer_state)

        if const_optim_state.dual_optimizer_state is not None:
            if dual_optimizer_class is None:
                raise ValueError(
                    "State dict contains dual_opt_state but dual_optimizer is None."
                )

            # This assumes a checkpoint-loaded formulation has been provided in
            # the initialization of the ``ConstrainedOptimizer``. This ensure
            # that we can safely call self.formulation.dual_parameters.
            dual_optimizer = dual_optimizer_class(formulation.dual_parameters)
            dual_optimizer.load_state_dict(const_optim_state.dual_optimizer_state)

            if const_optim_state.dual_scheduler_state is not None:
                if dual_scheduler_class is None:
                    raise ValueError(
                        "State dict contains dual_scheduler_state but dual_scheduler is None."
                    )

                dual_scheduler = dual_scheduler(dual_optimizer)
                dual_scheduler.load_state_dict(const_optim_state.dual_scheduler_state)
            else:
                dual_scheduler = None

        else:
            dual_optimizer = None
            dual_scheduler = None

        return ConstrainedOptimizer(
            formulation=formulation,
            primal_optimizer=primal_optimizer,
            dual_optimizer=dual_optimizer,
            dual_scheduler=dual_scheduler,
            alternating=const_optim_state.alternating,
            dual_restarts=const_optim_state.dual_restarts,
        )
