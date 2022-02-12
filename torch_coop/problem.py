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

    loss: torch.Tensor
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


# class ConstrainedMinimizationProblem(abc.ABC):
#     """Constrained minimization problem base class."""

#     def __init__(self):
#         self._state = None

#     @abc.abstractmethod
#     def loss(self):
#         """Returns loss function"""
#         pass

#     @property
#     @abc.abstractmethod
#     def is_constrained(self):
#         """Returns true if the problem is constrained"""
#         pass

#     @abc.abstractmethod
#     def ineq_constraints(self):
#         """Returns tensor inequality constraints"""
#         pass

#     @abc.abstractmethod
#     def eq_constraints(self):
#         """Returns tensor of equality constraints"""
#         pass

#     def proxy_ineq_constraints(self):
#         # Optional
#         return

#     def proxy_eq_constraints(self):
#         # Optional
#         return

#     @property
#     def state(self) -> ProblemState:
#         return self._state

#     @state.setter
#     def state(self, value: ProblemState):
#         self._state = value
#         # # Wastefully makes several function calls
#         # return ProblemState(
#         #     loss=self.loss(),
#         #     ineq_defect=self.ineq_constraints(),
#         #     eq_defect=self.eq_constraints(),
#         #     proxy_ineq_defect=self.proxy_ineq_constraints(),
#         #     proxy_eq_defect=self.proxy_eq_constraints(),
#         # )


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
