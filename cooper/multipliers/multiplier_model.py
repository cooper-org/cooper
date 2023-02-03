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
        Returns the *actual* value of the multipliers by
        passing the "features" of the constraint to predict the corresponding
        multiplier.
        """
        pass

    @property
    def shape(self):
        """
        Returns the shape of the explicit multipliers. In the case of implicit
        multipliers, this should return the *actual* predicted multipliers.
        """
        pass

    @property
    # FIXME(IsitaRex): Rename this.
    def grad(self) -> Iterator[torch.Tensor]:
        """Yields the current gradients stored in each fo the model parameters."""
        for parameter in self.parameters():
            yield parameter.grad

    def project_(self):
        raise RuntimeError("""project_ method does not exist for MultiplierModel.""")

    def restart_if_feasible_(self):
        raise RuntimeError(
            """restart_if_feasible_ method does not exist for MultiplierModel."""
        )
