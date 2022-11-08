from typing import Iterator

import torch
import abc
from cooper.multipliers import BaseMultiplier


class MultiplierModel(BaseMultiplier, metaclass=abc.ABCMeta):
    """
    A multiplier model. Holds a :py:class:`~torch.nn.Module`, which predicts
    the value of the Lagrange multipliers associated with the equality or
    inequality constraints of a
    :py:class:`~cooper.problem.ConstrainedMinimizationProblem`.

    Args:
        model: A :py:class:`~torch.nn.Module` which predicts the values of the
            Lagrange multipliers.
        is_positive: Whether to enforce non-negativity on the values of the
            multiplier.
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
    def grad(self):
        """Yields the current gradients stored in each fo the model parameters."""
        for param in self.parameters():
            yield param.grad

    def project_(self):
        raise RuntimeError("""project_ method does not exist for MultiplierModel.""")

    def restart_if_feasible_(self):
        raise RuntimeError(
            """restart_if_feasible_ method does not exist for MultiplierModel."""
        )

    # TODO: Add __str__ and similar methods to MultiplierModel if possible
