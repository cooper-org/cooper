from typing import List, Optional, Type, Union

import torch

from cooper.formulation import Formulation

from ..unconstrained_optimizer import UnconstrainedOptimizer
from .alternating_optimizer import AlternatingConstrainedOptimizer
from .cooper_optimizer import CooperOptimizer, CooperOptimizerState
from .extrapolation_optimizer import ExtrapolationConstrainedOptimizer
from .simultaneous_optimizer import SimultaneousConstrainedOptimizer


def create_optimizer_from_kwargs(
    formulation: Formulation,
    primal_optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer],
    dual_optimizer: Optional[torch.optim.Optimizer],
    dual_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    extrapolation: bool,
    alternating: bool,
    dual_restarts: bool,
) -> CooperOptimizer:
    """
    Create a CooperOptimizer from a set of keyword arguments. This method
    disambiguates the appropriate optimizer class to instantiate a new object.

    Args:
        formulation: CMP formulation.
        primal_optimizers: Fully instantiated optimizer(s) for the primal
            variables.
        dual_optimizer: Optional partially instantiated optimizer for the dual
            variables.
        dual_scheduler: Optional learning rate scheduler for the dual optimizer.
        extrapolation: Whether the optimizer uses extrapolation.
        alternating: Whether we perform alternating updates.
        dual_restarts: Whether we perform restarts on the dual variables.
    """

    if dual_optimizer is None:
        return UnconstrainedOptimizer(formulation=formulation, primal_optimizers=primal_optimizers)
    else:
        if extrapolation:
            return ExtrapolationConstrainedOptimizer(
                formulation=formulation,
                primal_optimizers=primal_optimizers,
                dual_optimizer=dual_optimizer,
                dual_scheduler=dual_scheduler,
                dual_restarts=dual_restarts,
            )

        elif alternating:
            return AlternatingConstrainedOptimizer(
                formulation=formulation,
                primal_optimizers=primal_optimizers,
                dual_optimizer=dual_optimizer,
                dual_scheduler=dual_scheduler,
                dual_restarts=dual_restarts,
            )

        elif dual_optimizer is not None:
            return SimultaneousConstrainedOptimizer(
                formulation=formulation,
                primal_optimizers=primal_optimizers,
                dual_optimizer=dual_optimizer,
                dual_scheduler=dual_scheduler,
                dual_restarts=dual_restarts,
            )


# Generate an appropriate CooperOptimizer based on a CooperOptimizerState.
def load_cooper_optimizer_from_state_dict(
    cooper_optimizer_state: CooperOptimizerState,
    formulation: Formulation,
    primal_optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer],
    dual_optimizer_class: Type[torch.optim.Optimizer] = None,
    dual_scheduler_class: Type[torch.optim.lr_scheduler._LRScheduler] = None,
):

    if isinstance(primal_optimizers, torch.optim.Optimizer):
        primal_optimizers = [primal_optimizers]

    primal_optimizer_states = cooper_optimizer_state.primal_optimizer_states
    if len(primal_optimizer_states) != len(primal_optimizers):
        raise ValueError(
            """The number of primal optimizers does not match the number of
            primal optimizer states."""
        )

    for optimizer, state in zip(primal_optimizers, primal_optimizer_states):
        optimizer.load_state_dict(state)

    if cooper_optimizer_state.dual_optimizer_state is not None:
        if dual_optimizer_class is None:
            raise ValueError("State dict contains dual_opt_state but dual_optimizer is None.")

        # This assumes a checkpoint-loaded formulation has been provided in
        # the initialization of the ``ConstrainedOptimizer``. This ensure
        # that we can safely call self.formulation.dual_parameters.
        dual_optimizer = dual_optimizer_class(formulation.dual_parameters)
        dual_optimizer.load_state_dict(cooper_optimizer_state.dual_optimizer_state)

        if cooper_optimizer_state.dual_scheduler_state is not None:
            if dual_scheduler_class is None:
                raise ValueError("State dict contains dual_scheduler_state but dual_scheduler is None.")

            dual_scheduler = dual_scheduler_class(dual_optimizer)
            dual_scheduler.load_state_dict(cooper_optimizer_state.dual_scheduler_state)
        else:
            dual_scheduler = None

    else:
        dual_optimizer = None
        dual_scheduler = None

    # Now we disambiguate the appropriate optimizer class to instantiate a new object

    optimizer_kwargs = {
        "formulation": formulation,
        "primal_optimizers": primal_optimizers,
        "dual_optimizer": dual_optimizer,
        "dual_scheduler": dual_scheduler,
        "extrapolation": cooper_optimizer_state.extrapolation,
        "alternating": cooper_optimizer_state.alternating,
        "dual_restarts": cooper_optimizer_state.dual_restarts,
    }

    return create_optimizer_from_kwargs(**optimizer_kwargs)
