import abc
from typing import Any, NamedTuple, Optional, TypedDict

import torch

from cooper.cmp import CMPState, ConstrainedMinimizationProblem, LagrangianStore
from cooper.utils import OneOrSequence, ensure_sequence


class CooperOptimizerState(TypedDict):
    r"""Stores the state of a :py:class:`~cooper.optim.CooperOptimizer`.

    Args:
        primal_optimizer_states: List of primal optimizer ``state_dict``\s.
        dual_optimizer_states: List of dual optimizer ``state_dict``\s. If the optimizer
            is an unconstrained optimizer, this field is set to ``None``.
    """

    primal_optimizer_states: list[dict]
    dual_optimizer_states: Optional[list[dict]]


class RollOut(NamedTuple):
    """Stores the output of a call to :py:meth:`~cooper.optim.CooperOptimizer.roll()`.

    Args:
        loss (:py:class:`torch.Tensor`): Value of the objective function.
        cmp_state (:py:class:`~cooper.cmp.CMPState`): State of the CMP.
        primal_lagrangian_store (:py:class:`~cooper.LagrangianStore`): LagrangianStore for the primal Lagrangian.
        dual_lagrangian_store (:py:class:`~cooper.LagrangianStore`): LagrangianStore for the dual Lagrangian.

    """

    loss: torch.Tensor
    cmp_state: CMPState
    primal_lagrangian_store: LagrangianStore
    dual_lagrangian_store: LagrangianStore


class CooperOptimizer(abc.ABC):
    r"""Base class for :py:class:`~cooper.optim.constrained_optimizer.ConstrainedOptimizer`
    and :py:class:`~cooper.optim.UnconstrainedOptimizer`\s.

    Args:
        cmp: The constrained minimization problem to be optimized. Providing the CMP
            as an argument for the constructor allows the optimizer to call the
            :py:meth:`~cooper.ConstrainedMinimizationProblem.compute_cmp_state`
            method within the :py:meth:`~cooper.optim.cooper_optimizer.CooperOptimizer.roll`
            method. Additionally, in the case of a constrained optimizer, the CMP
            enables access to the multipliers'
            :py:meth:`~cooper.multipliers.Multiplier.post_step_` method which must be
            called after the multiplier update.

        primal_optimizers: Optimizer(s) for the primal variables (e.g. the weights of
            a model). The primal parameters can be partitioned into multiple optimizers,
            in this case ``primal_optimizers`` accepts a list of
            :py:class:`torch.optim.Optimizer`\s.

        dual_optimizers: Optimizer(s) for the dual variables (e.g. the Lagrange
            multipliers associated with the constraints). A sequence of
            :py:class:`torch.optim.Optimizer`\s can be passed to handle the case of
            several :py:class:`~cooper.constraints.Constraint`\s.
    """

    def __init__(
        self,
        cmp: ConstrainedMinimizationProblem,
        primal_optimizers: OneOrSequence[torch.optim.Optimizer],
        dual_optimizers: Optional[OneOrSequence[torch.optim.Optimizer]] = None,
    ) -> None:
        self.cmp = cmp
        self.primal_optimizers = ensure_sequence(primal_optimizers)
        self.dual_optimizers = ensure_sequence(dual_optimizers)

    def zero_grad(self) -> None:
        r"""Sets the gradients of all optimized :py:class:`~torch.nn.parameter.Parameter`\s
        to zero. This includes both the primal and dual variables.
        """
        for primal_optimizer in self.primal_optimizers:
            # Prior to PyTorch 2.0, set_to_none=False was the default behavior.
            # The default behavior was changed to set_to_none=True in PyTorch 2.0.
            # We set set_to_none=True explicitly to ensure compatibility with both versions.
            primal_optimizer.zero_grad(set_to_none=True)

        if self.dual_optimizers is not None:
            for dual_optimizer in self.dual_optimizers:
                dual_optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def primal_step(self) -> None:
        """Performs a gradient step on the parameters associated with the primal variables."""
        for primal_optimizer in self.primal_optimizers:
            primal_optimizer.step()

    def state_dict(self) -> CooperOptimizerState:
        r"""Returns the state of the optimizer as a
        :py:class:`~cooper.optim.cooper_optimizer.CooperOptimizerState`. This method
        relies on the internal :py:meth:`~torch.optim.Optimizer.state_dict` method of
        the corresponding primal or dual optimizers.
        """
        primal_optimizer_states = [optimizer.state_dict() for optimizer in self.primal_optimizers]

        dual_optimizer_states = None
        if self.dual_optimizers is not None:
            dual_optimizer_states = [optimizer.state_dict() for optimizer in self.dual_optimizers]

        return CooperOptimizerState(
            primal_optimizer_states=primal_optimizer_states, dual_optimizer_states=dual_optimizer_states
        )

    def load_state_dict(self, state: CooperOptimizerState) -> None:
        """Loads the optimizer state from the given state dictionary.

        Args:
            state: A dictionary containing the optimizer state.

        Raises:
            ValueError: If the number of primal optimizers does not match the number of primal optimizer states.
            ValueError: If the number of dual optimizers does not match the number of dual optimizer states.
            ValueError: If ``dual_optimizer_states`` is present in the state dict but ``dual_optimizers`` is None.
        """
        if len(state["primal_optimizer_states"]) != len(self.primal_optimizers):
            raise ValueError("The number of primal optimizers does not match the number of primal optimizer states.")

        if self.dual_optimizers is None:
            if state["dual_optimizer_states"] is not None:
                raise ValueError(
                    "Optimizer state dict contains ``dual_optimizer_states`` but ``dual_optimizers`` is None."
                )
        elif len(state["dual_optimizer_states"]) != len(self.dual_optimizers):
            raise ValueError("The number of dual optimizers does not match the number of dual optimizer states.")

        for primal_optimizer, primal_optimizer_state in zip(self.primal_optimizers, state["primal_optimizer_states"]):
            primal_optimizer.load_state_dict(primal_optimizer_state)

        if self.dual_optimizers is not None:
            for dual_optimizer, dual_optimizer_state in zip(self.dual_optimizers, state["dual_optimizer_states"]):
                dual_optimizer.load_state_dict(dual_optimizer_state)

    @abc.abstractmethod
    def roll(self, *args: Any, **kwargs: Any) -> RollOut:
        """Evaluates the objective function and performs a gradient update on the parameters."""
