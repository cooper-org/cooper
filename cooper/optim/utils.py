from typing import Optional, Type, Union

import torch

from cooper.utils import OneOrSequence, ensure_sequence

from .. import ConstrainedMinimizationProblem
from . import constrained_optimizers
from .optimizer_state import CooperOptimizerState
from .unconstrained_optimizer import UnconstrainedOptimizer


def create_optimizer_from_kwargs(
    constrained_optimizers_class: Union[
        Type[constrained_optimizers.ConstrainedOptimizer], Type[UnconstrainedOptimizer]
    ],
    cmp: ConstrainedMinimizationProblem,
    primal_optimizers: OneOrSequence[torch.optim.Optimizer],
    dual_optimizers: Optional[OneOrSequence[torch.optim.Optimizer]] = None,
) -> Union[UnconstrainedOptimizer, constrained_optimizers.ConstrainedOptimizer]:
    """Creates a constrained or unconstrained optimizer from a set of keyword arguments."""

    if dual_optimizers is None:
        if constrained_optimizers_class != UnconstrainedOptimizer:
            raise ValueError("Dual optimizers must be provided for constrained optimization problems.")
        optimizer_kwargs = dict(primal_optimizers=primal_optimizers, cmp=cmp)
    else:
        optimizer_kwargs = dict(primal_optimizers=primal_optimizers, dual_optimizers=dual_optimizers, cmp=cmp)

    return constrained_optimizers_class(**optimizer_kwargs)


def load_cooper_optimizer_from_state_dict(
    constrained_optimizers_class: Union[
        Type[constrained_optimizers.ConstrainedOptimizer], Type[UnconstrainedOptimizer]
    ],
    cmp: ConstrainedMinimizationProblem,
    cooper_optimizer_state: CooperOptimizerState,
    primal_optimizers: OneOrSequence[torch.optim.Optimizer],
    dual_optimizers: Optional[OneOrSequence[torch.optim.Optimizer]] = None,
):
    """Creates a Cooper optimizer and loads the state_dicts contained in a
    :py:class:`~cooper.optim.CooperOptimizerState` onto instantiated primal and dual
    optimizers and constraints or multipliers.
    """

    # Load primal optimizers
    primal_optimizers = ensure_sequence(primal_optimizers)
    primal_optimizer_states = cooper_optimizer_state.primal_optimizer_states

    if len(primal_optimizer_states) != len(primal_optimizers):
        raise ValueError("The number of primal optimizers does not match the number of primal optimizer states.")

    for primal_optimizer, primal_state in zip(primal_optimizers, primal_optimizer_states):
        primal_optimizer.load_state_dict(primal_state)

    # Load dual optimizers
    dual_optimizer_states = cooper_optimizer_state.dual_optimizer_states
    if dual_optimizers is None:
        if dual_optimizer_states is not None:
            raise ValueError("Optimizer state dict contains `dual_optimizer_states` but `dual_optimizers` is None.")
    else:
        dual_optimizers = ensure_sequence(dual_optimizers)

        if dual_optimizer_states is None:
            raise ValueError("`dual_optimizers` were provided but state dict does not contain `dual_optimizer_states`.")

        if len(dual_optimizer_states) != len(dual_optimizers):
            raise ValueError("The number of dual optimizers does not match the number of dual optimizer states.")

        for dual_optimizer, dual_state in zip(dual_optimizers, dual_optimizer_states):
            dual_optimizer.load_state_dict(dual_state)

    return create_optimizer_from_kwargs(
        constrained_optimizers_class=constrained_optimizers_class,
        cmp=cmp,
        primal_optimizers=primal_optimizers,
        dual_optimizers=dual_optimizers,
    )
