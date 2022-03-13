Constrained Optimizer
=====================

.. currentmodule:: cooper.constrained_optimizer

.. autoclass:: ConstrainedOptimizer
    :members:


How to use a constrained optimizer
----------------------------------
The :py:class:`ConstrainedOptimizer` class is the cornerstone of **Cooper**. A
:py:class:`ConstrainedOptimizer` performs parameter updates to solve a
:py:class:`~cooper.problem.ConstrainedMinimizationProblem` given a chosen
:py:class:`~cooper.problem.Formulation`.

A ``ConstrainedOptimizer`` wraps a :py:class:`torch.optim.Optimizer`
used for updating the "primal" parameters associated directly with the
optimization problem. These might be, for example, the parameters of the model
you are training.

Additionally, a ``ConstrainedOptimizer`` includes a second
:py:class:`torch.optim.Optimizer`, which performs updates on the "dual"
parameters (e.g. the multipliers used in a
:py:class:`~cooper.lagrangian_formulation.LagrangianFormulation`).

Construction
^^^^^^^^^^^^

The main ingredients to build a ``ConstrainedOptimizer`` are a
:py:class:`~cooper.problem.Formulation` (associated with a
:py:class:`~cooper.problem.ConstrainedMinimizationProblem`) and a
:py:class:`torch.optim.Optimizer` corresponding to the ``primal_optimizer``.

If the ``ConstrainedMinimizationProblem`` you are dealing with is in fact
constrained, depending on your formulation, you might also need to provide a
``dual_optimizer``. Check out the section on :ref:`partial_optimizer_instantiation`
for more details on defining ``dual_optimizer``\s.


.. note::

    **Cooper** includes extra-gradient implementations of SGD and Adam which can
    be used as primal or dual optimizers. See :ref:`extra-gradient_optimizers`.

Examples:
~~~~~~~~~

The highlighted lines below show the small changes required to go from an
unconstrained to a constrained problem. Note that these changes should also be
accompanied with edits to the custom problem class which inherits from
:py:class:`~cooper.problem.ConstrainedMinimizationProblem`. More details on
the definition of a CMP can be found under the entry for :ref:`cmp`.

- **Unconstrained problem**

    .. code-block:: python
        :linenos:

        cmp = cooper.ConstrainedMinimizationProblem(is_constrained=False)
        formulation = cooper.problem.Formulation(...)

        primal_optimizer = cooper.optim.Adam(model.parameters(), lr=1e-2)

        const_optim = cooper.ConstrainedOptimizer(
            formulation=formulation,
            primal_optimizer=primal_optim,
        )

- **Constrained problem**

    .. code-block:: python
        :linenos:
        :emphasize-lines: 1,5,10

        cmp = cooper.ConstrainedMinimizationProblem(is_constrained=True)
        formulation = cooper.problem.Formulation(...)

        primal_optimizer = cooper.optim.Adam(model.parameters(), lr=1e-2)
        dual_optimizer = cooper.optim.partial(cooper.optim.SGD, lr=1e-3, momentum=0.9)

        const_optim = cooper.ConstrainedOptimizer(
            formulation=formulation,
            primal_optimizer=primal_optimizer,
            dual_optimizer=dual_optimizer,
        )

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


Additional features
-------------------


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
