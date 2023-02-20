from typing import List, Optional, Union

import torch

from cooper.constraints import ConstraintGroup
from cooper.utils import ensure_iterable

from ..unconstrained_optimizer import UnconstrainedOptimizer
from . import (
    AlternatingConstrainedOptimizer,
    ConstrainedOptimizer,
    CooperOptimizerState,
    ExtrapolationConstrainedOptimizer,
    SimultaneousConstrainedOptimizer,
)


def create_optimizer_from_kwargs(
    constraint_groups: Union[List[ConstraintGroup], ConstraintGroup],
    primal_optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer],
    dual_optimizers: Optional[Union[List[torch.optim.Optimizer], torch.optim.Optimizer]],
    extrapolation: bool,
    alternating: bool,
) -> Union[UnconstrainedOptimizer, ConstrainedOptimizer]:
    """
    Create a CooperOptimizer from a set of keyword arguments. This method
    disambiguates the appropriate optimizer class to instantiate a new object.

    Args:
        primal_optimizers: Fully instantiated optimizer(s) for the primal
            variables.
        dual_optimizer: Optional partially instantiated optimizer for the dual
            variables.
        extrapolation: Whether the optimizer uses extrapolation.
        alternating: Whether we perform alternating updates.
    """

    if dual_optimizers is None:
        return UnconstrainedOptimizer(constraint_groups=constraint_groups, primal_optimizers=primal_optimizers)

    if extrapolation:
        return ExtrapolationConstrainedOptimizer(
            constraint_groups=constraint_groups,
            primal_optimizers=primal_optimizers,
            dual_optimizers=dual_optimizers,
        )

    elif alternating:
        return AlternatingConstrainedOptimizer(
            constraint_groups=constraint_groups,
            primal_optimizers=primal_optimizers,
            dual_optimizers=dual_optimizers,
        )

    else:
        return SimultaneousConstrainedOptimizer(
            constraint_groups=constraint_groups,
            primal_optimizers=primal_optimizers,
            dual_optimizers=dual_optimizers,
        )


# Generate an appropriate CooperOptimizer based on a CooperOptimizerState.
def load_cooper_optimizer_from_state_dict(
    cooper_optimizer_state: CooperOptimizerState,
    constraint_groups: Union[List[ConstraintGroup], ConstraintGroup],
    primal_optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer],
    dual_optimizers: Optional[Union[List[torch.optim.Optimizer], torch.optim.Optimizer]],
):

    primal_optimizers = ensure_iterable(primal_optimizers)

    primal_optimizer_states = cooper_optimizer_state.primal_optimizer_states
    if len(primal_optimizer_states) != len(primal_optimizers):
        raise ValueError(
            """The number of primal optimizers does not match the number of
            primal optimizer states."""
        )

    for primal_optimizer, primal_state in zip(primal_optimizers, primal_optimizer_states):
        primal_optimizer.load_state_dict(primal_state)

    if dual_optimizers is None:
        if cooper_optimizer_state.dual_optimizer_state is not None:
            raise ValueError("State dict contains dual_opt_state but dual_optimizer is None.")

    else:
        dual_optimizer_states = cooper_optimizer_state.dual_optimizer_states
        dual_optimizers = ensure_iterable(dual_optimizers)

        if dual_optimizer_states is None:
            raise ValueError("State dict does not contain dual_optimizer_states but dual optimizers were provided.")

        if len(dual_optimizer_states) != len(dual_optimizers):
            raise ValueError(
                """The number of dual optimizers does not match the number of
                dual optimizer states."""
            )

        for dual_optimizer, dual_state in zip(dual_optimizers, dual_optimizer_states):
            dual_optimizer.load_state_dict(dual_state)

    # Now we disambiguate the appropriate optimizer class to instantiate a new object
    return create_optimizer_from_kwargs(
        constraint_groups=constraint_groups,
        primal_optimizers=primal_optimizers,
        dual_optimizers=dual_optimizers,
        extrapolation=cooper_optimizer_state.extrapolation,
        alternating=cooper_optimizer_state.alternating,
    )
