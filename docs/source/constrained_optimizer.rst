Constrained Optimizer
=====================

.. automodule:: cooper.constrained_optimizer

How to use a constrained optimizer
----------------------------------
To use :mod:`cooper.constrained_optimizer`, a ``ConstrainedOptimizer`` object
must be constructed. It will perform parameter updated to solve a
:py:class:`~cooper.problem.ConstrainedMinimizationProblem` given its
:py:class:`~cooper.problem.Formulation`.

It will hold a :py:class:`torch.optim.Optimizer` used on 'primal' parameters:
those associated directly with the optimization problem. It will potentially
also hold a :py:class:`torch.optim.Optimizer` used on 'dual' parameters, if
the provided formulation has such parameters (e.g. Lagrange multipliers).


Constructing it
---------------

Taking an optimization step
---------------------------

Base class
----------

.. currentmodule:: cooper.constrained_optimizer

.. autoclass:: ConstrainedOptimizer
    :members:
