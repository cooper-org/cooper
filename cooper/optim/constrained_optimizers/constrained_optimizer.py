# coding: utf8
"""
Implementation of the :py:class:`ConstrainedOptimizer` class.
"""

import dataclasses
import warnings
from collections.abc import Iterable
from typing import Dict, List, Optional, Union

import torch

from cooper.constraints import ConstraintGroup
from cooper.utils import ensure_iterable, validate_state_dicts


@dataclasses.dataclass
class CooperOptimizerState:
    """Represents the "state" of a Constrained Optimizer in terms of the state
    dicts of the primal optimizers, as well as those of the dual optimizer and
    the dual scheduler if applicable. This is used for checkpointing.

    Args:
        primal_optimizer_states: State dict for the primal optimizers.
        dual_optimizer_state: State dict for the dual optimizer.
        dual_scheduler_state: State dict for the dual scheduler.
    """

    primal_optimizer_states: List[Dict]
    dual_optimizer_states: Optional[List[Dict]] = None
    extrapolation: bool = False
    alternating: bool = False

    def __eq__(self, other):
        assert isinstance(other, CooperOptimizerState)

        def compare_state_dicts(dict_name):
            try:
                return validate_state_dicts(getattr(self, dict_name), getattr(other, dict_name))
            except:
                return False

        state_dict_names = ["primal_optimizer_states", "dual_optimizer_states"]
        all_checks = [compare_state_dicts(_) for _ in state_dict_names]

        all_checks.append(self.extrapolation == other.extrapolation)
        all_checks.append(self.alternating == other.alternating)

        return all(all_checks)

    def asdict(self):
        return dataclasses.asdict(self)


class ConstrainedOptimizer:

    """
    Optimizes a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`
    given a provided :py:class:`~cooper.formulation.Formulation`.

    A ``ConstrainedOptimizer`` includes one or more
    :class:`torch.optim.Optimizer`\\s for the primal variables. It also includes
    a :class:`torch.optim.Optimizer` for the dual variables associated with the
    provided ``Formulation``.

    For handling unconstrained problems, we provide an
    :py:class:`~cooper.optim.cooper_optimizer.UnconstrainedOptimizer`. Please
    refer to the documentation of the
    :py:class:`~cooper.problem.ConstrainedMinimizationProblem` and
    :py:class:`~cooper.formulation.Formulation` classes for further details on
    handling unconstrained problems.

    Args:
        constraint_groups:

        primal_optimizers: Fully instantiated ``torch.optim.Optimizer``\\s used
            to optimize the primal parameters (e.g. model parameters). The primal
            parameters can be partitioned into multiple optimizers, in this case
            ``primal_optimizers`` accepts a list of ``torch.optim.Optimizer``\\s.

        dual_optimizer: Partially instantiated ``torch.optim.Optimizer``
            used to optimize the dual variables (e.g. Lagrange multipliers).
            If dealing with an unconstrained problem, please use a
            :py:class:`~cooper.optim.cooper_optimizer.UnconstrainedOptimizer`
            instead.

        dual_scheduler: Partially instantiated
            ``torch.optim.lr_scheduler._LRScheduler`` used to schedule the
            learning rate of the dual variables. Defaults to None.

        extrapolation: Whether to perform extragradient updates. Defaults to False.

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
        constraint_groups: Union[List[ConstraintGroup], ConstraintGroup],
        primal_optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer],
        dual_optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer],
    ):
        self.constraint_groups = ensure_iterable(constraint_groups)
        self.primal_optimizers = ensure_iterable(primal_optimizers)
        self.dual_optimizers = ensure_iterable(dual_optimizers)

    def base_sanity_checks(self):
        """
        Perform sanity checks on the initialization of ``ConstrainedOptimizer``.
        """

        if len(self.constraint_groups) != len(self.dual_optimizers):
            raise RuntimeError("One dual optimizer must be provided for each constraint group.")

        if self.constraint_groups is None:
            raise RuntimeError("No constraints were provided.")

        if self.primal_optimizers is None:
            raise RuntimeError("No primal optimizer was provided.")

        if self.dual_optimizers is None:
            raise RuntimeError("No dual optimizer was provided.")

        any_restart_on_feasible = False
        any_is_augmented_lagrangian = False
        for constraint in self.constraint_groups:
            if hasattr(constraint.multiplier.restart_on_feasible) and constraint.multiplier.restart_on_feasible:
                any_restart_on_feasible = True
            if constraint.formulation.formulation_type == "augmented_lagrangian":
                any_is_augmented_lagrangian = True

        if self.alternating and any_restart_on_feasible:
            warnings.warn(
                """Using alternating updates with dual restarts is untested.
                Please use with caution."""
            )

        if any_is_augmented_lagrangian and not self.alternating:
            raise RuntimeError("Augmented Lagrangian formulation requires alternating updates.")

    def zero_grad(self):
        """
        Sets the gradients of all optimized
        :py:class:`~torch.nn.parameter.Parameter`\\s to zero. This includes both
        the primal and dual variables.
        """
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.zero_grad()

        for dual_optimizer in self.dual_optimizers:
            dual_optimizer.zero_grad()

    def state_dict(self) -> CooperOptimizerState:
        """
        Returns the state of the ConstrainedOptimizer. See
        :py:class:`~cooper.optim.constrained_optimizers.cooper_optimizer.CooperOptimizerState`.
        """

        primal_optimizer_states = [_.state_dict() for _ in self.primal_optimizers]
        dual_optimizer_states = [_.state_dict() for _ in self.dual_optimizers]

        return CooperOptimizerState(
            primal_optimizer_states=primal_optimizer_states,
            dual_optimizer_states=dual_optimizer_states,
            extrapolation=self.extrapolation,
            alternating=self.alternating,
        )
