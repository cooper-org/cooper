#!/usr/bin/env python

"""Classes for modeling dual variables (e.g. Lagrange multipliers)."""

import torch


class BaseMultiplier(torch.nn.Module):
    """
    Placeholder for base multiplier class, which can be extended to adapt
    different types of multipliers: Dense, Sparse, implicit multiplier which is
    itself the output of a model.

    .. todo::
        Implement BaseMultiplier class and refactor DenseMultiplier.
    """

    def __init__(self) -> None:
        super().__init__()


class DenseMultiplier(torch.nn.Module):
    """
    A dense multiplier. Holds a :py:class:`~torch.nn.parameter.Parameter`,
    which contains the value of the Lagrange multipliers associated with the
    equality or inequality constraints of a
    :py:class:`~cooper.problem.ConstrainedMinimizationProblem`.

    Args:
        init: Initial value of the multiplier.
        positive: Whether to enforce non-negativity on the values of the
            multiplier.

    Attributes:
        shape: shape of the :py:class:`~torch.nn.parameter.Parameter` associated
            with the multiplier.
        data: data (weight) of the :py:class:`~torch.nn.parameter.Parameter`
            associated with the multiplier.
        grad: gradient of the :py:class:`~torch.nn.parameter.Parameter`
            associated with the multiplier.
    """

    def __init__(self, init: torch.Tensor, *, positive: bool = False):
        super().__init__()
        self.weight = torch.nn.Parameter(init)
        self.positive = positive

    @property
    def shape(self):
        return self.weight.shape

    @property
    def grad(self):
        return self.weight.grad

    @property
    def data(self):
        return self.weight.data

    @data.setter
    def data(self, value):
        self.weight.data = value

    def forward(self):
        """
        Defines the computation performed at every call, which gets the current
        value of the multiplier.
        """
        return self.weight

    def project_(self):
        """
        Generic projection for non-negative multipliers used in inequality
        constraints. May be generalized to other custom projections.
        """
        if self.positive:
            self.weight.data = torch.relu(self.data)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        pos_str = "inequality" if self.positive else "equality"
        rep = "DenseMultiplier(" + pos_str + ", " + str(self.data) + ")"
        return rep
