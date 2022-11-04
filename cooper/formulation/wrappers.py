import torch

from .formulation import Formulation


def compute_lagrangian(formulation: Formulation, *args, **kwargs) -> torch.Tensor:
    """Alias for :py:meth:`._composite_objective`"""
    return formulation._composite_objective(*args, **kwargs)


def backward(formulation: Formulation, lagrangian: torch.Tensor):
    """Alias for :py:meth:`._populate_gradients` to keep the  ``backward``
    naming convention used in Pytorch. For clarity, we avoid naming this
    method ``backward`` as it is a method of the ``LagrangianFormulation``
    object and not a method of a :py:class:`torch.Tensor` as is standard in
    Pytorch.
    """
    formulation._populate_gradients(lagrangian)
