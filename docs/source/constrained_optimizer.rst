Constrained Optimizer
=====================

.. automodule:: cooper.constrained_optimizer

How to use a constrained optimizer
----------------------------------
To use ``cooper.constrained_optimizer``, a ``ConstrainedOptimizer`` object
must be constructed. It will perform parameter updates to solve a
:py:class:`~cooper.problem.ConstrainedMinimizationProblem` given its
:py:class:`~cooper.problem.Formulation`.

It will hold a :py:class:`torch.optim.Optimizer` used on "primal" parameters:
those associated directly with the optimization problem. It will potentially
also hold a :py:class:`torch.optim.Optimizer` for "dual" parameters, if
the provided formulation has them (e.g. Lagrange multipliers).

Constructing it
^^^^^^^^^^^^^^^

To construct a ``ConstrainedOptimizer``, you have to give it a
``Formulation`` of a ``ConstrainedMinimizationProblem``
and an instantiated ``Optimizer`` for the primal (e.g. model) parameters.

If the ``ConstrainedMinimizationProblem`` you are trying to solve is constrained,
you must also provide a partially instantiated ``Optimizer`` for the dual
parameters (see :py:func:`cooper.optim.partial`). This partialy instantiated
class must only lack the ``params`` attribute, which will be created at runtime
by the ``Formulation`` based on the properties of evaluated constraint defects.

.. note::

    In particular, :py:class:`cooper.optim.Extragradient` optimizers with an
    extrapolation method can be provided for the primal and dual parameters.

Examples:

- Unconstrained problem::

    cmp = cooper.ConstrainedMinimizationProblem(is_constrained=False)
    formulation = ...
    primal_optim = cooper.optim.Adam(model.parameters(), lr=1e-2)

    const_optim = cooper.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optim,
        )

- Constrained problem::

    cmp = cooper.ConstrainedMinimizationProblem(is_constrained=True)
    formulation = ...
    primal_optim = cooper.optim.Adam(model.parameters(), lr=1e-2)
    p_dual_optim = cooper.optim.partial(cooper.optim.SGD, lr=1e-3, momentum=0.9)

    const_optim = cooper.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optim,
        dual_optimizer=p_dual_optim,
        )

Alternating updates
~~~~~~~~~~~~~~~~~~~

You can also indicate ``alternating=True`` for updates to alternate between the
primal and dual parameters. In this case, the gradient you compute by calling
:py:meth:`~cooper.problem.Formulation.custom_backward` is used to update the
primal parameters and is then discarded. The gradient with respect to the dual
variables must then be re-computed to update them, process which is handled
internally in :py:meth:`ConstrainedOptimizer.step`.

.. note::

    Choosing ``alternating=True`` does not necessarily double the number of
    backward passes through a model. When ``LagrangianFormulation`` calculates
    gradients with respect to :py:class:`~cooper.multipliers.DenseMultiplier`\s,
    it sufices to evaluate the constraint defects (through a call to
    :py:meth:`~cooper.problem.ConstrainedMinimizationProblem.closure`), which is
    cheaper than having to backpropagate through the Lagrangian.

Dual restarts
~~~~~~~~~~~~~

``dual_restarts=True`` can be used to set the dual variables associated with
inequality constraints to 0 each time that their respective constraint reaches
feasibility. In practice, this prevents the optimization from over-focusing on
the constraints at the cost of the loss function.

We recommend to set ``dual_restarts=False`` when dealing with
constraints whose violations are estimated stochastically, for
example Monte Carlo estimates for expectations. This is to avoid restarting
multipliers when a constraint is being satisfied for **one** estimation, as
it may not be satisfied for further estimations because of their stochasticity.

Taking an optimization step
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``ConstrainedOptimizer`` implements a :meth:`~ConstrainedOptimizer.step`
method, that updates the primal and dual parameters (if ``Formulation`` has any).
The nature of the update depends on the attributes provided during the
initialization of the ``ConstrainedOptimizer``. By default, updates are via
gradient descent on the primal parameters and (projected) ascent
on the dual parameters, with simultaneous updates.

Other methods for solving a ``ConstrainedMinimizationProblem`` include:

    - Projected gradient descent-ascent with alternating updates.
    - Extra-gradient.
    - Augmented Lagrangian.

.. note::

    When solving an unconstrained problem, ``ConstrainedOptimizer.step()`` will
    perform usual gradient descent updates on the primal parameters (given the primal
    optimizer provided). This is done as the notions of a Lagrangian and a
    saddle point optimization problem do not arise in the unconstrained case.

.. warning::

    Only one choice of the mentioned optimization methods can be used at a time.
    See :meth:`ConstrainedOptimizer.sanity_checks` for more details.

The :meth:`~ConstrainedOptimizer.step` method can be used in one of two ways:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. ``constrained_optimizer.step()``

    This is the basic usage of the method, which applies to gradient descent-acent
    and augmented Lagrangian updates. The function can be called once the gradients
    are computed using :py:meth:`~cooper.problem.Formulation.custom_backward`.

#. ``constrained_optimizer.step(closure)``

    Some optimization algorithms such as extragradient and alternating gradient
    descent-ascent need to re-evaluate the state of the ``ConstrainedMinimizationProblem``
    multiple times, so you have to pass in the
    :py:meth:`~cooper.problem.ConstrainedMinimizationProblem.closure` function
    that allows us to recompute the :py:class:`~cooper.problem.CMPState` and
    call :py:meth:`~cooper.problem.Formulation.custom_backward` based on it.
    The arguments required to evaluate ``closure()`` must also be provided as
    ``*closure_args`` and ``**closure_kwargs``.

Example::

    cmp = cooper.ConstrainedMinimizationProblem(...)
    formulation = cooper.LagrangianFormulation(...)
    constrained_optimizer = cooper.ConstrainedOptimizer(...)

    for batch, targets in dataset:
        constrained_optimizer.zero_grad()

        # The closure is always required to compute the Lagrangian
        lagrangian = formulation.composite_objective(cmp.closure, *closure_args, **closure_kwargs)
        formulation.custom_backward(lagrangian)

        # Not providing a closure
        constrained_optimizer.step()

        # Providing closure
        constrained_optimizer.step(cmp.closure, *closure_args, **closure_kwargs)

Base class
----------

.. currentmodule:: cooper.constrained_optimizer

.. autoclass:: ConstrainedOptimizer
    :members:
