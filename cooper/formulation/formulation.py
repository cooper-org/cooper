import abc
from typing import Any, Callable, Dict, Optional, Union

import torch

from cooper.problem import CMPState, ConstrainedMinimizationProblem

# from .lagrangian_model import CMPModelState


# Formulation, and some other classes below, are heavily inspired by the design
# of the TensorFlow Constrained Optimization (TFCO) library :
# https://github.com/google-research/tensorflow_constrained_optimization


class Formulation(abc.ABC):
    """Base class for formulations of CMPs."""

    def __init__(self, cmp: Optional[ConstrainedMinimizationProblem] = None):
        self.cmp = cmp

    @abc.abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass

    @abc.abstractmethod
    def create_state(self):
        """Initializes the internal state of the formulation."""
        pass

    @abc.abstractmethod
    def state(self):
        """Returns the internal state of formulation (e.g. multipliers)."""
        pass

    @abc.abstractmethod
    def flip_dual_gradients(self):
        """Flips the sign of the dual gradients."""
        pass

    @property
    @abc.abstractmethod
    def is_state_created(self):
        """Returns ``True`` if the internal state has been created."""
        pass

    @property
    @abc.abstractmethod
    def dual_parameters(self):
        """Returns the trainable parameters for the dual variables. Depending on
        the formulation, these dual parameters can represent the multipliers
        themselves, or a model which "learns" multiplier values."""
        pass

    @abc.abstractmethod
    def compute_lagrangian(self):
        pass

    @abc.abstractmethod
    def backward(self, *args, **kwargs):
        """Performs the backward computation and populates the gradients
        for the primal and dual variables according to the design of the
        formulation."""
        pass

    # TODO(daoterog): fix circular import type hint can be correct
    def write_cmp_state(self, cmp_state: CMPState):  # Union[CMPState, CMPModelState]):
        """Provided that the formulation is linked to a
        `ConstrainedMinimizationProblem`, writes a CMPState to the CMP."""

        if self.cmp is None:
            raise RuntimeError(
                """Cannot write state to a formulation which is not linked to a
                ConstrainedMinimizationProblem"""
            )

        self.cmp.state = cmp_state


class UnconstrainedFormulation(Formulation):
    """
    Base class for unconstrained formulations.

    Attributes:
        cmp: :py:class:`~cooper.problem.ConstrainedMinimizationProblem` we aim
            to solve and which gives rise to the Lagrangian.
    """

    def __init__(self, cmp: Optional[ConstrainedMinimizationProblem] = None):
        """Construct new `UnconstrainedFormulation`"""

        self.cmp = cmp

    def create_state(self):
        """This is a stateless formulation. No need to create a state."""
        pass

    def state(self) -> None:
        """Returns the internal state of formulation (e.g. multipliers)."""
        return None

    @property
    def is_state_created(self) -> False:
        """This is a stateless formulation. This function always returns ``False``."""
        return False

    @property
    def dual_parameters(self) -> None:
        """Returns ``None`` as there are no trainable dual parameters in an
        unconstrained formulation."""
        return None

    def state_dict(self) -> Dict[str, Any]:
        """
        Generates the state dictionary for an unconstrained formulation.
        """

        return {}

    def load_state_dict(self, state_dict: dict):
        """
        Loads the state dictionary for an unconstrained formulation. Since
        unconstrained formulations are stateless, this is a no-op.
        """
        pass

    def flip_dual_gradients(self):
        """Flips the sign of the dual gradients. This is a no-op for
        unconstrained formulations."""
        pass

    def compute_lagrangian(
        self,
        closure: Callable[..., CMPState],
        *closure_args,
        write_state: Optional[bool] = True,
        **closure_kwargs
    ) -> torch.Tensor:
        """
        Computes the loss based on a new evaluation of the
        :py:class:`~cooper.problem.CMPState``.

        Args:
            closure: Callable returning a :py:class:`cooper.problem.CMPState`
            write_state: If ``True``, the ``state`` of the formulation's
                :py:class:`cooper.problem.ConstrainedMinimizationProblem`
                attribute is replaced by that returned by the ``closure``
                argument. This flag can be used (when set to ``False``) to
                evaluate the loss, e.g. for logging validation metrics,
                without overwritting the information stored in the formulation's
                :py:class:`cooper.problem.ConstrainedMinimizationProblem`.
        """

        cmp_state = closure(*closure_args, **closure_kwargs)

        if write_state and self.cmp is not None:
            self.write_cmp_state(cmp_state)

        return cmp_state.loss

    def backward(self, loss: torch.Tensor):
        """
        Performs the backward computation which populates the gradients for the
        primal variables.

        Args:
            loss: Loss tensor for computing gradients for primal variables.
        """
        loss.backward()
