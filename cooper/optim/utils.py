"""Utility functions for partial instantiation of dual optimizers and schedulers"""

import functools
from typing import Type, no_type_check

import torch
from torch.optim.lr_scheduler import _LRScheduler


@no_type_check
def partial_optimizer(optim_cls: Type[torch.optim.Optimizer], **optim_kwargs):
    """
    Partially instantiates an optimizer class. This approach is preferred over
    :py:func:`functools.partial` since the returned value is an optimizer
    class whose attributes can be inspected and which can be further
    instantiated.

    Args:
        optim_cls: Pytorch optimizer class to be partially instantiated.
        **optim_kwargs: Keyword arguments for optimizer hyperparemeters.
    """

    class PartialOptimizer(optim_cls):
        __init__ = functools.partialmethod(optim_cls.__init__, **optim_kwargs)

    return PartialOptimizer


@no_type_check
def partial_scheduler(scheduler_cls: Type[_LRScheduler], **scheduler_kwargs):
    """
    Partially instantiates a learning rate scheduler class. This approach is
    preferred over :py:func:`functools.partial` since the returned value is a
    scheduler class whose attributes can be inspected and which can be further
    instantiated.

    Args:
        scheduler_cls: Pytorch scheduler class to be partially instantiated.
        **scheduler_kwargs: Keyword arguments for scheduler hyperparemeters.
    """

    class PartialScheduler(scheduler_cls):
        __init__ = functools.partialmethod(scheduler_cls.__init__, **scheduler_kwargs)

    return PartialScheduler
