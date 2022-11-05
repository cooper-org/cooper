import torch
import abc
from cooper.multipliers import BaseMultiplier


class MultiplierModel(BaseMultiplier, meta=abc.ABCMeta):
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

    @property
    def grad(self):
        """Yields the current gradients stored in each fo the model parameters."""
        for param in self.model.parameters():
            if param.grad is not None:
                yield param.grad

    @abc.abstractmethod
    def forward(self, constraint_features: torch.Tensor):
        """
        Returns the *actual* value of the multipliers by
        passing the "features" of the constraint to predict the corresponding
        multiplier.
        """
        pass

    def __str__(self):
        return str(self.model.input.data)

    def __repr__(self):
        pos_str = "inequality" if self.positive else "equality"
