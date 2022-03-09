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
^^^^^^^^^^^^^^^

To construct a ``ConstrainedOptimizer``, you have to give it a
:py:class:`~cooper.problem.Formulation` of a ``ConstrainedMinimizationProblem``
and an instantiated ``Optimizer`` for the primal (e.g. model) parameters.

If the ``ConstrainedMinimizationProblem`` you are trying to solve is constrained,
you must also provide a partially instantiated ``Optimizer`` for the dual
parameters (see :py:func:`cooper.optim.partial`). This instantiated class must
only lack the ``params`` attribute, which will be created at runtime by the
``Formulation`` based on evaluations of constraint defects.

.. note::

    Dual parameters are internally moved to the same device as the primal
    parameters.

Examples:

- Unconstrained problem::

    cmp = cooper.problem.ConstrainedMinimizationProblem(is_constrained=False)
    formulation = ...
    primal_optim = cooper.optim.Adam(model.parameters(), lr=1e-2)

    const_optim = cooper.constrained_optimizer.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optim,
        )

- Constrained problem::

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``ConstrainedOptimizer`` implements a :func:`~cooper.ConstrainedOptimizer.step`
method, that updates the primal and dual parameters if any. The nature of the
update depends on the attributes provided during the initialization of the
``ConstrainedOptimizer``. By default, updates are via simultaneous projected
gradient descent on the primal parameters and ascent on the dual parameters.

Other methods for solving the `ConstrainedMinimizationProblem` include:

    - Alternating projected gradient descent-ascent.
    - Extrapolation.
    - Augmented Lagrangian.

.. note::

    When solving an unconstrained problem, the notions of a Lagrangian and a
    saddle point optimization do not exist. Therefore, updates will be gradient
    descent on the primal parameters (given the primal optimizer provided).

The :func:`~cooper.ConstrainedOptimizer.step` method can be used in one of two
ways:

``optimizer.step()``
~~~~~~~~~~~~~~~~~~~~

This is the basic usage of the method, which applies to gradient descent-acent
and augmented Lagrangian updates. The function can be called once the gradients
are computed using :py:meth:`~cooper.problem.Formulation.custom_backward`.

Example::

    cmp = ...
    formulation = ...
    constrained_optimizer = ...

    for batch, targets in dataset:
        constrained_optimizer.zero_grad()
        lagrangian = formulation.composite_objective(cmp.closure, *closure_args, **closure_kwargs)
        formulation.custom_backward(lagrangian)
        constrained_optimizer.step()

``optimizer.step(closure)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some optimization algorithms such as Extragradient and alternating gradient
descent-ascent need to re-evaluate the state of the ``ConstrainedMinimizationProblem``
multiple times, so you have to pass in the closure that allows them to recompute
your :py:class:`~cooper.problem.CMPState`. The arguments required to evaluate the
closure must also be provided as ``*closure_args`` and ``**closure_kwargs``.

Example::

    cmp = ...
    formulation = ...
    constrained_optimizer = ...

    for batch, targets in dataset:
        constrained_optimizer.zero_grad()
        lagrangian = formulation.composite_objective(cmp.closure, *closure_args, **closure_kwargs)
        formulation.custom_backward(lagrangian)
        constrained_optimizer.step(cmp.closure, *closure_args, **closure_kwargs)

Base class
----------

.. currentmodule:: cooper.constrained_optimizer

.. autoclass:: ConstrainedOptimizer
    :members:
