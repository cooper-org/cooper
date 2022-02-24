import abc
from dataclasses import dataclass
from typing import Optional

import torch

# Formulation, and some other classes below, are heavily inspired by the design
# of the TensorFlow Constrained Optimization (TFCO) library :
# https://github.com/google-research/tensorflow_constrained_optimization


class ConstrainedMinimizationProblem(abc.ABC):
    """Constrained minimization problem base class."""

    loss: Optional[torch.Tensor] = None
    ineq_defect: Optional[torch.Tensor] = None
    eq_defect: Optional[torch.Tensor] = None
    proxy_ineq_defect: Optional[torch.Tensor] = None
    proxy_eq_defect: Optional[torch.Tensor] = None
    misc: Optional[dict] = None

    def __init__(self, is_constrained=False):
        self.is_constrained = is_constrained

    @property
    def state(self) -> tuple:
        return (
            self.loss,
            self.ineq_defect,
            self.eq_defect,
            self.proxy_ineq_defect,
            self.proxy_eq_defect,
            self.misc,
        )

    @abc.abstractmethod
    def update_state(self, *cmp_args, **cmp_kwargs):
        """Defined by the user. Sets the state of the problem given args
        and kwargs."""
        pass


class Formulation(abc.ABC):
    """Base class for Lagrangian and proxy-Lagrangian formulations"""

    def __init__(self):
        self.cmp = None

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
    def composite_objective(self, *cmp_args, **closure_kwargs):
        pass

    @abc.abstractmethod
    def populate_gradients(self):
        """Like lagrangian backward"""
        pass
