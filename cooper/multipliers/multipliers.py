"""Classes for modeling dual variables (e.g. Lagrange multipliers)."""
import abc

import torch


class ConstantMultiplier:
    """
    Constant (non-trainable) multiplier class used for penalized formulations.

    Args:
        init: Value of the multiplier.
    """

    def __init__(self, init: torch.Tensor):
        if init.requires_grad:
            raise ValueError("Constant multiplier should not be trainable.")
        self.weight = init
        self.device = init.device

    def __call__(self):
        """Return the current value of the multiplier."""
        return self.weight

    def state_dict(self):
        return {"weight": self.weight}

    def load_state_dict(self, state_dict):
        self.weight = state_dict["weight"]


class ExplicitMultiplier(torch.nn.Module):
    """
    A dense multiplier. Holds a :py:class:`~torch.nn.parameter.Parameter`, which
    contains the value of the Lagrange multipliers associated with the equality or
    inequality constraints of a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`.

    Args:
        init: Initial value of the multiplier.
        positive: Whether to enforce non-negativity on the values of the multiplier.
    """

    def __init__(self, init: torch.Tensor, *, enforce_positive: bool = False):
        super().__init__()
        self.enforce_positive = enforce_positive

        if self.enforce_positive and any(init < 0):
            raise ValueError("For inequality constraint, all entries in multiplier must be non-negative.")

        self.weight = torch.nn.Parameter(init)
        self.device = self.weight.device

    def project_(self):
        """
        Ensures non-negativity for multipliers associated with inequality constraints.
        """
        if self.positive:
            self.weight.relu_()

    @property
    def implicit_constraint_type(self):
        return "ineq" if self.enforce_positive else "eq"

    def restart_if_feasible_(self, feasible_indices: torch.Tensor, restart_value: float = 0.0):
        """
        In-place restart function for multipliers.

        Args:
            feasible_indices: Indices or binary masks denoting the feasible constraints.
        """

        if not self.enforce_positive:
            raise RuntimeError("Restarts are only supported for inequality multipliers")

        self.weight.data[feasible_indices, ...] = restart_value
        if self.weight.grad is not None:
            self.weight.grad[feasible_indices, ...] = 0.0

    def state_dict(self):
        _state_dict = super().state_dict()
        _state_dict["enforce_positive"] = self.enforce_positive
        return _state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.enforce_positive = state_dict["enforce_positive"]
        self.device = self.weight.device


class DenseMultiplier(ExplicitMultiplier):
    def forward(self):
        """Return the current value of the multiplier."""
        return self.weight

    def __repr__(self):
        return f"DenseMultiplier({self.implicit_constraint_type}, shape={self.weight.shape})"


class SparseMultiplier(ExplicitMultiplier):
    def forward(self, indices: torch.Tensor):
        """Return the current value of the multiplier at the provided indices."""
        return torch.nn.functional.embedding(indices, self.weight, sparse=True).squeeze()

    def __repr__(self):
        return f"SparseMultiplier({self.implicit_constraint_type}, shape={self.weight.shape})"


class ImplicitMultiplier(torch.nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self):
        pass
