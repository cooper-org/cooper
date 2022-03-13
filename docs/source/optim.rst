Optim
=====================

.. currentmodule:: cooper.optim


This module gathers aliases for :py:class:`torch.optim.Optimizer`\s so they
can be accessed directly from **Cooper** as ``cooper.optim.<OptimizerName>``. The
existing aliases are:

- ``cooper.optim.SGD = torch.optim.SGD``
- ``cooper.optim.Adam = torch.optim.Adam``
- ``cooper.optim.Adagrad = torch.optim.Adagrad``
- ``cooper.optim.RMSprop = torch.optim.RMSprop``

.. _partial_optimizer_instantiation:

Partial optimizer instantiation
-------------------------------

When constructing a :py:class:`~cooper.constrained_optimizer.ConstrainedOptimizer`, the
``dual_optimizer`` parameter is expected to be a
:py:class:`torch.optim.Optimizer` for which the ``params`` argument has **not
yet** been passed. The rest of the instantiation of the ``dual_optimizer`` is
handled internally by **Cooper**.

The :py:meth:`cooper.optim.partial` method below allows you to provide a
configuration for your ``dual_optimizer``\'s hyperparameters (e.g. learning
rate, momentum, etc.)

.. automethod:: cooper.optim.partial


.. _extra-gradient_optimizers:

Extra-gradient optimizers
-------------------------

.. autoclass:: ExtraSGD
    :members:

.. autoclass:: ExtraAdam
    :members:
