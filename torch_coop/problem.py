import abc
from dataclasses import dataclass
from typing import Optional

import torch


# Formulation, and some other classes below, are heavily inspired by the design
# of the TensorFlow Constrained Optimization (TFCO) library :
# https://github.com/google-research/tensorflow_constrained_optimization


@dataclass
class CMPState:
    """Represents the "state" of a Constrained Minimization Problem in terms of
    the value of its loss and constraint violations/defects.
    """

    loss: Optional[torch.Tensor] = None
    ineq_defect: Optional[torch.Tensor] = None
    eq_defect: Optional[torch.Tensor] = None
    proxy_ineq_defect: Optional[torch.Tensor] = None
    proxy_eq_defect: Optional[torch.Tensor] = None

    def as_tuple(self) -> tuple:
        return (
            self.loss,
            self.ineq_defect,
            self.eq_defect,
            self.proxy_ineq_defect,
            self.proxy_eq_defect,
        )


class ConstrainedMinimizationProblem(abc.ABC):
    """Constrained minimization problem base class."""

    def __init__(self, is_constrained=False):
        self.is_constrained = is_constrained
        self._state = None

    @property
    def state(self) -> CMPState:
        return self._state

    @state.setter
    def state(self, value: CMPState):
        self._state = value


class Formulation(abc.ABC):
    """Base class for Lagrangian and proxy-Lagrangian formulations"""

    @abc.abstractmethod
    def state(self):
        """Returns internal state of formulation (e.g. multipliers)"""
        pass

    @abc.abstractmethod
    def create_state(self):
        """Initializes the internal state/multipliers"""
        pass

    @property
    @abc.abstractmethod
    def is_state_created(self):
        """Returns True if the internal state has been created"""
        pass

    @property
    @abc.abstractmethod
    def dual_parameters(self):
        pass

    @abc.abstractmethod
    def get_composite_objective(self, cmp):
        """Closure-like function"""
        pass

    @abc.abstractmethod
    def populate_gradients(self):
        """Like lagrangian backward"""
        pass
