Constrained Optimizer
=====================

.. automodule:: cooper.constrained_optimizer

How to use a constrained optimizer
----------------------------------
To use :mod:`cooper.constrained_optimizer`, a ``ConstrainedOptimizer`` object
must be constructed. It will perform parameter updates to solve a
:py:class:`~cooper.problem.ConstrainedMinimizationProblem` given its
:py:class:`~cooper.problem.Formulation`.

It will hold a :py:class:`torch.optim.Optimizer` used on 'primal' parameters:
those associated directly with the optimization problem. It will potentially
also hold a :py:class:`torch.optim.Optimizer` used on 'dual' parameters, if
the provided formulation has them (e.g. Lagrange multipliers).

Constructing it
---------------

To construct a ``ConstrainedOptimizer``, you have to give it a
:py:class:`~cooper.problem.Formulation` of a ``ConstrainedMinimizationProblem``
and an instantiated ``Optimizer`` for the primal (e.g. model) parameters.

If the ``ConstrainedMinimizationProblem`` you are trying to solve is constrained,
you must also provide a partially instantiated ``Optimizer`` for the dual
parameters (see :py:meth:`cooper.optim.partial`). This instantiated class must
only lack the ``params`` attribute, which will be created at runtime by the
``Formulation`` based on the evaluated constraint defects.

.. note::

    Dual parameters are internally moved to the same device as the primal
    parameters.

Example: unconstrained problem::

    # Unconstrained problem
    cmp = cooper.problem.ConstrainedMinimizationProblem(is_constrained=False)
    formulation = ...
    primal_optim = cooper.optim.Adam(model.parameters(), lr=1e-2)

    const_optim = cooper.constrained_optimizer.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optim,
        )

Example: constrained problem::

    # Constrained problem
    cmp = cooper.problem.ConstrainedMinimizationProblem(is_constrained=True)
    formulation = ...
    primal_optim = cooper.optim.Adam(model.parameters(), lr=1e-2)
    p_dual_optim = cooper.optim.partial(cooper.optim.SGD, lr=1e-3, momentum=0.9)

    const_optim = cooper.constrained_optimizer.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optim,
        dual_optimizer=p_dual_optim,
        )


Taking an optimization step
---------------------------

Base class
----------

.. currentmodule:: cooper.constrained_optimizer

.. autoclass:: ConstrainedOptimizer
    :members:
