import abc
from typing import Iterator

import torch

from .multipliers import BaseMultiplier


class MultiplierModel(BaseMultiplier, metaclass=abc.ABCMeta):
    """
    A multiplier model. Holds a :py:class:`~torch.nn.Module`, which predicts
    the value of the Lagrange multipliers associated with the equality or
    inequality constraints of a
    :py:class:`~cooper.problem.ConstrainedMinimizationProblem`. This is class is meant
    to be inherited by the user to implement their own multiplier model.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, constraint_features: torch.Tensor):
        """
        Returns the *actual* value of the multipliers by passing the "features" of the
        constraint to predict the corresponding multiplier.
        """
        pass

    @property
    def grad(self) -> Iterator[torch.Tensor]:
        raise RuntimeError("""grad method does not exist for MultiplierModel.""")

    @property
    def shape(self):
        raise RuntimeError("""shape method does not exist for MultiplierModel.""")

    def project_(self):
        raise RuntimeError("""project_ method does not exist for MultiplierModel.""")

    def restart_if_feasible_(self):
        raise RuntimeError(
            """restart_if_feasible_ method does not exist for MultiplierModel."""
        )
