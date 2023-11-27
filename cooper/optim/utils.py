import warnings
from enum import Enum
from typing import Optional, Union

import torch

from cooper.constraints import ConstraintGroup
from cooper.multipliers import Multiplier
from cooper.utils import OneOrSequence, ensure_sequence

from . import constrained_optimizers
from .optimizer_state import CooperOptimizerState
from .types import AlternationType
from .unconstrained_optimizer import UnconstrainedOptimizer


def create_optimizer_from_kwargs(
    primal_optimizers: OneOrSequence[torch.optim.Optimizer],
    extrapolation: bool = False,
    alternation_type: AlternationType = AlternationType.FALSE,
    dual_optimizers: Optional[OneOrSequence[torch.optim.Optimizer]] = None,
    multipliers: Optional[OneOrSequence[Multiplier]] = None,
) -> Union[UnconstrainedOptimizer, constrained_optimizers.ConstrainedOptimizer]:
    """Creates a constrained or unconstrained optimizer from a set of keyword arguments.
    This method disambiguates the appropriate optimizer class to instantiate.

    Args:
        primal_optimizers: Optimizer(s) for the primal variables.
        extrapolation: Whether the optimizer uses extrapolation.
        alternation_type: Choice of alternation strategy.
        dual_optimizer: Optional optimizer(s) for the dual variables.
    """

    if dual_optimizers is None:
        return UnconstrainedOptimizer(primal_optimizers=primal_optimizers)

    optimizer_kwargs = dict(
        primal_optimizers=primal_optimizers,
        dual_optimizers=dual_optimizers,
        multipliers=multipliers,
    )

    if extrapolation:
        return constrained_optimizers.ExtrapolationConstrainedOptimizer(**optimizer_kwargs)
    elif alternation_type == AlternationType.PRIMAL_DUAL:
        return constrained_optimizers.AlternatingPrimalDualOptimizer(**optimizer_kwargs)
    elif alternation_type == AlternationType.DUAL_PRIMAL:
        return constrained_optimizers.AlternatingDualPrimalOptimizer(**optimizer_kwargs)
    else:
        return constrained_optimizers.SimultaneousOptimizer(**optimizer_kwargs)


def load_cooper_optimizer_from_state_dict(
    cooper_optimizer_state: CooperOptimizerState,
    primal_optimizers: OneOrSequence[torch.optim.Optimizer],
    dual_optimizers: Optional[OneOrSequence[torch.optim.Optimizer]] = None,
    multipliers: Optional[OneOrSequence[Multiplier]] = None,
):
    """Creates a Cooper optimizer and loads the state_dicts contained in a
    :py:class:`~cooper.optim.CooperOptimizerState` onto instantiated primal and dual
    optimizers and constraint groups or multipliers.
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

    # Load multipliers
    multipliers = ensure_sequence(multipliers) if multipliers is not None else []

    multiplier_states = cooper_optimizer_state.multiplier_states
    if multiplier_states is None:
        if len(multipliers) > 0:
            raise ValueError("Unable to load multiplier states since state dict does not contain `multiplier_states`.")
    else:
        if len(multiplier_states) != len(multipliers):
            raise ValueError("The number of multipliers does not match the number of multiplier states.")
        for multiplier, multiplier_state in zip(multipliers, multiplier_states):
            multiplier.load_state_dict(multiplier_state)

    # Since we have extracted the multiplier information above, we discard the constraint_groups below
    return create_optimizer_from_kwargs(
        primal_optimizers=primal_optimizers,
        extrapolation=cooper_optimizer_state.extrapolation,
        alternation_type=cooper_optimizer_state.alternation_type,
        dual_optimizers=dual_optimizers,
        multipliers=multipliers,
    )
