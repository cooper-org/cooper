#!/usr/bin/env python

"""Classes for modeling dual variables (e.g. Lagrange multipliers)."""

import abc

import torch


class BaseMultiplier(torch.nn.Module, metaclass=abc.ABCMeta):
    """
    Base class for Lagrange multipliers. This base class can be extended to
    different types of multipliers: Dense, Sparse or implicit multipliers.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    @abc.abstractmethod
    def shape(self):
        """
        Returns the shape of the explicit multipliers. In the case of implicit
        multipliers, this should return the *actual* predicted multipliers.
        """
        pass

    @property
    @abc.abstractmethod
    def grad(self):
        """
        Returns the gradient of trainable parameters associated with the
        multipliers. In the case of implicit multipliers, this corresponds to
        the gradient with respect to the parameters of the model which predicts
        the multiplier values.
        """
        pass

    @abc.abstractmethod
    def forward(self):
        """
        Returns the *actual* value of the multipliers. When using implicit
        multipliers, the signature of this method may be change to enable
        passing the "features" of the constraint to predict the corresponding
        multiplier.
        """
        pass

    @abc.abstractmethod
    def project_(self):
        """
        In-place projection function for multipliers.
        """
        pass

    @abc.abstractmethod
    def restart_if_feasible_(self):
        """
        In-place restart function for multipliers.
        """
        pass


class DenseMultiplier(BaseMultiplier):
    """
    A dense multiplier. Holds a :py:class:`~torch.nn.parameter.Parameter`,
    which contains the value of the Lagrange multipliers associated with the
    equality or inequality constraints of a
    :py:class:`~cooper.problem.ConstrainedMinimizationProblem`.

    Args:
        init: Initial value of the multiplier.
        positive: Whether to enforce non-negativity on the values of the
            multiplier.
    """

    def __init__(self, init: torch.Tensor, *, positive: bool = False):
        super().__init__()
        self.weight = torch.nn.Parameter(init)
        self.positive = positive
        self.device = self.weight.device

    @property
    def shape(self):
        """Returns the shape of the multiplier tensor."""
        return self.weight.shape

    @property
    def grad(self):
        """Returns current gradient stored in the multiplier tensor."""
        return self.weight.grad

    def forward(self):
        """Return the current value of the multiplier."""
        return self.weight

    def project_(self):
        """
        Ensures multipliers associated with inequality constraints reamain
        non-negative.
        """
        if self.positive:
            self.weight.data = torch.relu(self.weight.data)

    def restart_if_feasible_(self):
        """
        In-place restart function for multipliers.
        """

        assert self.positive, "Restarts is only supported for inequality multipliers"

        # Call to formulation.backwards has already flipped sign
        # A currently *positive* gradient means original defect is negative, so
        # the constraint is being satisfied.

        # The code below still works in the case of proxy constraints, since the
        # multiplier updates are computed based on *non-proxy* constraints
        feasible_filter = self.weight.grad > 0
        self.weight.grad[feasible_filter] = 0.0
        self.weight.data[feasible_filter] = 0.0

        self.weight.grad[feasible_filter] = 0.0
        self.weight.data[feasible_filter] = 0.0

    def __str__(self):
        return str(self.weight.data)

    def __repr__(self):
        pos_str = "inequality" if self.positive else "equality"
        rep = "DenseMultiplier(" + pos_str + ", " + str(self.weight.data) + ")"
        return rep

    def __members(self):
        return (self.positive, self.weight)

    def __hash__(self):
        return hash(self.__members())

    def __eq__(self, other):

        if type(other) is type(self):

            self_positive, self_weight = self.__members()
            other_positive, other_weight = other.__members()

            positive_check = self_positive == other_positive
            weight_check = torch.allclose(self_weight, other_weight)

            return positive_check and weight_check

        else:
            return False
