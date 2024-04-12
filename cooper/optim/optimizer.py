import abc
from typing import NamedTuple, Optional, TypedDict

import torch

from cooper.cmp import CMPState, ConstrainedMinimizationProblem, LagrangianStore
from cooper.utils import OneOrSequence, ensure_sequence


class CooperOptimizerState(TypedDict):
    primal_optimizer_states: list[dict]
    dual_optimizer_states: Optional[list[dict]]


class RollOut(NamedTuple):
    """Stores the output of a call to :py:meth:`~cooper.optim.cooper_optimizer.CooperOptimizer.roll`."""

    loss: torch.Tensor
    cmp_state: CMPState
    primal_lagrangian_store: LagrangianStore
    dual_lagrangian_store: LagrangianStore


class CooperOptimizer(abc.ABC):
    def __init__(
        self,
        cmp: ConstrainedMinimizationProblem,
        primal_optimizers: OneOrSequence[torch.optim.Optimizer],
        dual_optimizers: Optional[OneOrSequence[torch.optim.Optimizer]] = None,
    ):
        self.cmp = cmp
        self.primal_optimizers = ensure_sequence(primal_optimizers)
        self.dual_optimizers = ensure_sequence(dual_optimizers) if dual_optimizers is not None else None

    def zero_grad(self):
        """
        Sets the gradients of all optimized :py:class:`~torch.nn.parameter.Parameter`\\s
        to zero. This includes both the primal and dual variables.
        """
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.zero_grad()

        if self.dual_optimizers is not None:
            for dual_optimizer in self.dual_optimizers:
                dual_optimizer.zero_grad()

    def primal_step(self):
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.step()

    def state_dict(self) -> CooperOptimizerState:
        primal_optimizer_states = [optimizer.state_dict() for optimizer in self.primal_optimizers]

        dual_optimizer_states = None
        if self.dual_optimizers is not None:
            dual_optimizer_states = [optimizer.state_dict() for optimizer in self.dual_optimizers]

        return CooperOptimizerState(
            primal_optimizer_states=primal_optimizer_states, dual_optimizer_states=dual_optimizer_states
        )

    def load_state_dict(self, state: CooperOptimizerState):
        if len(state["primal_optimizer_states"]) != len(self.primal_optimizers):
            raise ValueError("The number of primal optimizers does not match the number of primal optimizer states.")

        if self.dual_optimizers is None:
            if state["dual_optimizer_states"] is not None:
                raise ValueError("Optimizer state dict contains `dual_optimizer_states` but `dual_optimizers` is None.")
        else:
            if len(state["dual_optimizer_states"]) != len(self.dual_optimizers):
                raise ValueError("The number of dual optimizers does not match the number of dual optimizer states.")

        for primal_optimizer, primal_optimizer_state in zip(self.primal_optimizers, state["primal_optimizer_states"]):
            primal_optimizer.load_state_dict(primal_optimizer_state)

        if self.dual_optimizers is not None:
            for dual_optimizer, dual_optimizer_state in zip(self.dual_optimizers, state["dual_optimizer_states"]):
                dual_optimizer.load_state_dict(dual_optimizer_state)

    @abc.abstractmethod
    def roll(self, *args, **kwargs) -> RollOut:
        """Evaluates the objective function and performs a gradient update on the parameters."""
        raise NotImplementedError
