Constrained Optimizer
=====================

.. currentmodule:: cooper.constrained_optimizer

.. autoclass:: ConstrainedOptimizer
    :members:


How to use a ``ConstrainedOptimizer``
-------------------------------------

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

Examples
~~~~~~~~~

The highlighted lines below show the small changes required to go from an
unconstrained to a constrained problem. Note that these changes should also be
accompanied with edits to the custom problem class which inherits from
:py:class:`~cooper.problem.ConstrainedMinimizationProblem`. More details on
the definition of a CMP can be found under the entry for :ref:`cmp`.


- **Unconstrained problem**

    .. code-block:: python
        :linenos:

        model =  ModelClass(...)
        cmp = cooper.ConstrainedMinimizationProblem(is_constrained=False)
        formulation = cooper.problem.Formulation(...)

        primal_optimizer = cooper.optim.Adam(model.parameters(), lr=1e-2)

        constrained_optimizer = cooper.ConstrainedOptimizer(
            formulation=formulation,
            primal_optimizer=primal_optim,
        )

- **Constrained problem**

    .. code-block:: python
        :linenos:
        :emphasize-lines: 2,7,12

        model =  ModelClass(...)
        cmp = cooper.ConstrainedMinimizationProblem(is_constrained=True)
        formulation = cooper.problem.Formulation(...)

        primal_optimizer = cooper.optim.Adam(model.parameters(), lr=1e-2)
        # Note that dual_optimizer is "partly instantiated", *without* parameters
        dual_optimizer = cooper.optim.partial(cooper.optim.SGD, lr=1e-3, momentum=0.9)

        constrained_optimizer = cooper.ConstrainedOptimizer(
            formulation=formulation,
            primal_optimizer=primal_optimizer,
            dual_optimizer=dual_optimizer,
        )

The training loop
^^^^^^^^^^^^^^^^^

We have gathered all the ingredients we need for tackling our CMP: the
custom :py:class:`~cooper.problem.ConstrainedMinimizationProblem` class, along
with your :py:class:`ConstrainedOptimizer` of choice and a
:py:class:`ConstrainedOptimizer` for updating the parameters. Now it is time to
put them to good use.

The typical training loop for solving a CMP in a machine learning set up using
**Cooper** (with a :ref:`Lagrangian Formulation<lagrangian_formulations>`)
will involve the following steps:

.. admonition:: Overview of main steps in a training loop
    :class: hint

    #. (Optional) Iterate over your dataset and sample of mini-batch.
    #. Call :py:meth:`constrained_optimizer.zero_grad()<zero_grad>` to reset the parameters' gradients
    #. Compute the current :py:class:`CMPState` (or estimate it with the minibatch) and calculate the Lagrangian using :py:meth:`lagrangian.composite_objective(cmp.closure, ...)<cooper.lagrangian_formulation.LagrangianFormulation.composite_objective>`.
    #. Populate the primal and dual gradients with :py:meth:`formulation.custom_backward(lagrangian)<cooper.problem.Formulation.custom_backward>`
    #. Perform updates on the parameters using the primal and dual optimizers based on the recently computed gradients, via a call to :py:meth:`constrained_optimizer.step()<step>`.

Example
^^^^^^^

    .. code-block:: python
        :linenos:

        model = ModelClass(...)
        cmp = cooper.ConstrainedMinimizationProblem(...)
        formulation = cooper.LagrangianFormulation(...)

        primal_optimizer = cooper.optim.SGD(model.parameters(), lr=primal_lr)
        # Note that dual_optimizer is "partly instantiated", *without* parameters
        dual_optimizer = cooper.optim.partial(cooper.optim.SGD, lr=primal_lr)

        constrained_optimizer = cooper.ConstrainedOptimizer(
            formulation=formulation,
            primal_optimizer=primal_optimizer,
            dual_optimizer=dual_optimizer,
        )

        for inputs, targets in dataset:
            # Clear gradient buffers
            constrained_optimizer.zero_grad()

            # The closure is required to compute the Lagrangian
            # The closure might in turn require the model, inputs, targets, etc.
            lagrangian = formulation.composite_objective(cmp.closure, ...)

            # Populate the primal and dual gradients
            formulation.custom_backward(lagrangian)

            # Perform primal and dual parameter updates
            constrained_optimizer.step()


Parameter updates
^^^^^^^^^^^^^^^^^

By default, parameter updates will be performed using **simultaneous** gradient
descent-ascent updates (according to the choice of primal and dual optimizers).
Formally,

.. math::

    x_{t+1} &= \texttt{primal_optimizer_update} \left( x_{t}, \nabla_{x} \mathcal{L}(x_t, \lambda_t) \right)\\
    \lambda_{t+1} &= \texttt{dual_optimizer_update} \left( \lambda_{t}, {\color{red} \mathbf{-}} \nabla_{\lambda} \mathcal{L}(x_t, \lambda_t) \right)

.. note::
    We explicitly include a negative sign in front of the gradient for
    :math:`\lambda` in order to highlight the fact that :math:`\lambda` solves
    **maximization** problem. **Cooper** handles the sign flipping internally, so
    you should provide your definition for a ``dual_optimizer`` using a non-negative
    learning rate, as usual!

.. admonition:: Multiplier projection
    :class: note

    Lagrange multipliers associated with inequality constraints should remain
    non-negative. **Cooper** executes the standard projection to
    :math:`\mathbb{R}^{+}` by default for
    :py:class:`~cooper.optim.multipliers.DenseMultiplier`\s. For more details
    on using custom projection operations, see the section on :ref:`multipliers`.


Other update strategies implemented by :py:class:`~ConstrainedOptimizer`
include:

- (TODO) :ref:`Alternating updates<alternating_updates>` for (projected) gradient descent-ascent
- Performing :ref:`dual_restarts` on the Lagrange multipliers for inequality constraints
- Using :ref:`Extra-gradient<extra-gradient_optimizers>`

  - Extra-gradient-based optimizers require an extra call to the
    :py:meth:`cmp.closure()<cooper.problem.ConstrainedMinimizationProblem.closure>`.
    See the section on :ref:`extra-gradient_optimizers` for usage details.
- TODO: :ref:`augmented_lagrangian`

The ``ConstrainedOptimizer`` implements a :py:meth:`ConstrainedOptimizer.step`
method, that updates the primal and dual parameters (if ``Formulation`` has any).
The nature of the update depends on the attributes provided during the
initialization of the ``ConstrainedOptimizer``. By default, updates are via
gradient descent on the primal parameters and (projected) ascent
on the dual parameters, with simultaneous updates.


.. note::

    When applied to an unconstrained problem, :py:meth:`ConstrainedOptimizer.step`
    will be equivalent to performing ``primal_optimizer.step()`` based on the
    gradient of the loss with respect to the primal parameters.

Additional features
-------------------

.. _alternating_updates:

Alternating updates
^^^^^^^^^^^^^^^^^^^

It is possible to perform alternating updates between the primal and dual
parameters by setting the flag ``alternating=True`` in the construction of the
:py:class:`ConstrainedOptimizer`. In this case, the gradient computed by calling
:py:meth:`~cooper.problem.Formulation.custom_backward` is used to update the
primal parameters. Then, the gradient with respect to the dual variables (given
the new value of the primal parameter!) is computed and used to update the dual
variables. This two-stage process is handled by **Cooper** inside the
:py:meth:`ConstrainedOptimizer.step` method.

.. math::

    x_{t+1} &= \texttt{primal_optimizer_update} \left( x_{t}, \nabla_{x} \mathcal{L}(x_t, \lambda_t) \right)\\
    \lambda_{t+1} &= \texttt{dual_optimizer_update} \left( \lambda_{t}, {\color{red} \mathbf{-}} \nabla_{\lambda} \mathcal{L}({\color{red} x_{t+1}}, \lambda_t) \right)


.. important::

    Selecting ``alternating=True`` does not necessarily double the number of
    backward passes through a the primal parameters!

    When using a ``LagrangianFormulation``, to obtain the gradients with respect
    to the Lagrange multipliers, it suffices to *evaluate* the constraint
    defects (through a call to
    :py:meth:`~cooper.problem.ConstrainedMinimizationProblem.closure`). This
    operation does not require a having to back-propagate through the Lagrangian.

.. todo::

    This functionality is untested and is yet to be integrated with the use of
    proxy constraints.

.. _dual_restarts:

Dual restarts
^^^^^^^^^^^^^

``dual_restarts=True`` can be used to set the value of the dual variables
associated with inequality constraints to zero whenever the respective
constraints are being satisfied. We do not perform ``dual_restarts`` on
multipliers for equality constraints.

For simplicity, consider a CMP with two inequality constraints
:math:`g_1(x) \le 0` and :math:`g_2(x) \le 0` and loss :math:`f(x)`. Suppose
that the constraint on :math:`g_1` is strictly satisfied, while that on
:math:`g_2` is violated at :math:`x`. Consider the Lagrangian at :math:`x`
with (**non-negative**) multipliers :math:`\lambda_1` and :math:`\lambda_2`.

.. math::
    \mathcal{L}(x,\lambda_1, \lambda_2) = f(x) + \lambda_1 \, g_1(x) + \lambda_2 \, g_2(x)

The `best response <https://en.wikipedia.org/wiki/Best_response>`_ for
:math:`\lambda_1` and :math:`\lambda_2` at the current value of the primal
parameters :math:`x` is :math:`(\lambda_1, \lambda_2) = (0, +\infty)`. This is
due to the non-negativity constraints on the Lagrange multipliers, the sign of
the constraint violations, and the fact that the :math:`\lambda`-player wants to
maximize :math:`\mathcal{L}(x,\lambda_1, \lambda_2)`.

"Playing a best response" for the Lagrange multiplier :math:`\lambda_2` of the
violated constraint clearly leads to an impractical algorithm (due to numerical
overflow). However, for the currently *feasible* constraint, performing a best
response update on :math:`\lambda_1` is indeed implementable, and trivially so.
This is exactly the effect of setting ``dual_restarts=True``!

In practice, this prevents the optimization from over-focusing on a constraint
(at the expense on improving on the loss function!) when said constraint **is
already being satisfied**. This over-penalization arises from the accumulation
of the (positive) defects in the value of the Lagrange multiplier while the
constraint was being violated in the past.

.. note::
    We recommend setting ``dual_restarts=False`` when dealing with constraints
    whose violations are estimated stochastically, for example Monte Carlo
    estimates constraints described by expectations.  This is to avoid
    restarting multipliers when a constraint is "being satisfied" for a single
    estimate, since this can be a result of the stochasticity of the estimator.

.. _augmented_lagrangian:

Augmented Lagrangian
^^^^^^^^^^^^^^^^^^^^

.. todo::

    Verify current (unfinished and untested) implementation of the Augmented
    Lagrangian method.
