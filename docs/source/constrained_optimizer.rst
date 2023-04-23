Constrained Optimizer
=====================

.. currentmodule:: cooper.optim.constrained_optimizers.constrained_optimizer

How to use a ``ConstrainedOptimizer``
-------------------------------------

The :py:class:`ConstrainedOptimizer` class is the cornerstone of **Cooper**. A
:py:class:`ConstrainedOptimizer` performs parameter updates to solve a
:py:class:`~cooper.problem.ConstrainedMinimizationProblem` given a chosen
:py:class:`~cooper.formulation.Formulation`.

A ``ConstrainedOptimizer`` wraps a :py:class:`torch.optim.Optimizer`
used for updating the "primal" parameters associated directly with the
optimization problem. These might be, for example, the parameters of the model
you are training.

Additionally, a ``ConstrainedOptimizer`` includes a second
:py:class:`torch.optim.Optimizer`, which performs updates on the "dual"
parameters (e.g. the multipliers used in a
:py:class:`~cooper.formulation.LagrangianFormulation`).

Construction
^^^^^^^^^^^^

The main ingredients to build a ``ConstrainedOptimizer`` are a
:py:class:`~cooper.formulation.Formulation` (associated with a
:py:class:`~cooper.problem.ConstrainedMinimizationProblem`) and a
:py:class:`torch.optim.Optimizer` corresponding to a ``primal_optimizer``.

.. note::
    **Cooper** supports the use of multiple ``primal_optimizers``, each
    corresponding to different groups of primal variables. The
    ``primal_optimizers`` argument accepts a single optimizer, or a list
    of optimizers. See :ref:`multiple-primal_optimizers`.

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
        cmp = cooper.ConstrainedMinimizationProblem()
        formulation = cooper.formulation.Formulation(...)

        primal_optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        constrained_optimizer = cooper.UnconstrainedOptimizer(
            formulation=formulation,
            primal_optimizers=primal_optimizer,
        )

- **Constrained problem**

    .. code-block:: python
        :linenos:
        :emphasize-lines: 7,9,12

        model =  ModelClass(...)
        cmp = cooper.ConstrainedMinimizationProblem()
        formulation = cooper.formulation.Formulation(...)

        primal_optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        # Note that dual_optimizer is "partly instantiated", *without* parameters
        dual_optimizer = cooper.optim.partial_optimizer(torch.optim.SGD, lr=1e-3, momentum=0.9)

        constrained_optimizer = cooper.SimultaneousOptimizer(
            formulation=formulation,
            primal_optimizers=primal_optimizer,
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
    #. Compute the current :py:class:`CMPState` (or estimate it with the minibatch) and calculate the Lagrangian using :py:meth:`formulation.compute_lagrangian(cmp.closure, ...)<cooper.formulation.LagrangianFormulation.compute_lagrangian>`.
    #. Populate the primal and dual gradients with :py:meth:`formulation.backward(lagrangian)<cooper.formulation.Formulation.backward>`
    #. Perform updates on the parameters using the primal and dual optimizers based on the recently computed gradients, via a call to :py:meth:`constrained_optimizer.step()<step>`.

Example
~~~~~~~

    .. code-block:: python
        :linenos:

        model = ModelClass(...)
        cmp = cooper.ConstrainedMinimizationProblem(...)
        formulation = cooper.LagrangianFormulation(...)

        primal_optimizer = torch.optim.SGD(model.parameters(), lr=primal_lr)
        # Note that dual_optimizer is "partly instantiated", *without* parameters
        dual_optimizer = cooper.optim.partial_optimizer(torch.optim.SGD, lr=primal_lr)

        constrained_optimizer = cooper.ConstrainedOptimizer(
            formulation=formulation,
            primal_optimizers=primal_optimizer,
            dual_optimizer=dual_optimizer,
        )

        for inputs, targets in dataset:
            # Clear gradient buffers
            constrained_optimizer.zero_grad()

            # The closure is required to compute the Lagrangian
            # The closure might in turn require the model, inputs, targets, etc.
            lagrangian = formulation.compute_lagrangian(cmp.closure, ...)

            # Populate the primal and dual gradients
            formulation.backward(lagrangian)

            # Perform primal and dual parameter updates
            constrained_optimizer.step()

.. _basic_parameter_updates:

Parameter updates
^^^^^^^^^^^^^^^^^

By default, parameter updates are performed using **simultaneous** gradient
descent-ascent updates (according to the choice of primal and dual optimizers).
Formally,


.. math::

    x_{t+1} &= \texttt{primal_optimizer_update} \left( x_{t}, \nabla_{x} \mathcal{L}(x, \lambda_t)|_{x=x_t} \right)\\
    \lambda_{t+1} &= \texttt{dual_optimizer_update} \left( \lambda_{t}, {\color{red} \mathbf{-}} \nabla_{\lambda} \mathcal{L}({x_{t}}, \lambda)|_{\lambda=\lambda_t} \right)

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


Other update strategies supported by :py:class:`~ConstrainedOptimizer` include:

- :ref:`Alternating updates<alternating_updates>` for (projected) gradient descent-ascent
- The :ref:`Augmented Lagrangian<augmented_lagrangian_const_opt>` method (ALM)
- Performing :ref:`dual_restarts` on the Lagrange multipliers for inequality constraints
- Using :ref:`Extra-gradient<extra-gradient_optimizers>`

  - Extra-gradient-based optimizers require an extra call to the
    :py:meth:`cmp.closure()<cooper.problem.ConstrainedMinimizationProblem.closure>`.
    See the section on :ref:`extra-gradient_optimizers` for usage details.

The ``ConstrainedOptimizer`` implements a :py:meth:`ConstrainedOptimizer.step`
method, that updates the primal and dual parameters (if ``Formulation`` has any).
The nature of the update depends on the attributes provided during the
initialization of the ``ConstrainedOptimizer``. By default, updates are via
gradient descent on the primal parameters and (projected) ascent
on the dual parameters, with simultaneous updates.


.. note::

    When applied to an unconstrained problem, :py:meth:`ConstrainedOptimizer.step`
    will be equivalent to performing ``optimizer.step()`` on all of the
    ``primal_optimizers`` based on the gradient of the loss with respect to the
    primal parameters.


.. include:: additional_features.rst

``ConstrainedOptimizer`` Class
------------------------------

.. autoclass:: ConstrainedOptimizer
    :members:
