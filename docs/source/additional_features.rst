.. currentmodule:: cooper.optim.constrained_optimizers

Additional features
-------------------

In this section we provide details on using "advanced features" such as
alternating updates, or the Augmented Lagrangian method, in conjunction with a
:py:class:`~cooper.optim.constrained_optimizers.ConstrainedOptimizer`.

--------------------------------------------------------------------------------

.. _alternating_updates:

Alternating updates
^^^^^^^^^^^^^^^^^^^

It is possible to perform alternating updates between the primal and dual
parameters via the ``alternation_type`` argument in the construction of the
:py:class:`ConstrainedOptimizer`. In this case, the gradient computed by calling
:py:meth:`~cooper.formulation.Formulation.backward` is used to update the
primal parameters. Then, the gradient with respect to the dual variables (given
the new value of the primal parameters!) is computed and used to update the dual
variables. This two-stage process is handled by **Cooper** inside the
:py:meth:`ConstrainedOptimizer.step` method.


One can perform alternating updates in which the primal parameters are updated first. We
refer to this update strategy as ``cooper.optim.AlternationType.PRIMAL_DUAL``.
.. math::

    x_{t+1} &= \texttt{primal_optimizers_update} \left( x_{t}, \nabla_{x} \mathcal{L}_{c_t}(x, \lambda_t)|_{x=x_t} \right)\\
    \lambda_{t+1} &= \texttt{dual_optimizer_update} \left( \lambda_{t}, {\color{red} \mathbf{-}} \nabla_{\lambda} \mathcal{L}({\color{red} x_{t+1}}, \lambda)|_{\lambda=\lambda_t} \right)

Alternative, ``cooper.optim.AlternationType.DUAL_PRIMAL`` carries out an update of the
dual parameters first.
.. math::

    \lambda_{t+1} &= \texttt{dual_optimizer_update} \left( \lambda_{t}, {\color{red} \mathbf{-}} \nabla_{\lambda} \mathcal{L}({\color{red} x_{t}}, \lambda)|_{\lambda=\lambda_t} \right) \\
    x_{t+1} &= \texttt{primal_optimizers_update} \left( x_{t}, \nabla_{x} \mathcal{L}_{c_t}(x, \lambda_{t+1})|_{x=x_t} \right)

.. important::

    Selecting ``alternation_type=AlternationType.DualPrimal`` does not double the number
    of backward passes through the primal parameters!

    When using a ``LagrangianFormulation``, to obtain the gradients with respect
    to the Lagrange multipliers, it suffices to *evaluate* the constraint
    defects (through a call to
    :py:meth:`~cooper.problem.ConstrainedMinimizationProblem.closure`). This
    operation does not require a having to back-propagate through the Lagrangian
    with respect to the primal parameters.

    Providing a ``defect_fn`` in the call to :py:meth:`ConstrainedOptimizer.step`
    allows for updating the Lagrange multiplier without having to re-evaluate
    the loss function, but rather only the constraints.

--------------------------------------------------------------------------------

.. _augmented_lagrangian_const_opt:

Augmented Lagrangian method
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Exactly solving the intermediate optimization problems considered by the
:ref:`Augmented Lagrangian formulation<augmented_lagrangian_formulation>` can
be challenging. This is particularly true when :math:`L_{c_t}(x, \lambda)` is
non-convex in :math:`x`.

Rather than aiming to solve the intermediate optimization problem to high
precision, **Cooper** implements a version of the Augmented Lagrangian method
where the primal variables are updated using a gradient-based step.

.. math::
    x_{t+1} = \texttt{primal_optimizers_update} \left( x_{t}, \nabla_{x} \mathcal{L}_{c_t}(x, \lambda_t)|_{x=x_t} \right)

The new primal variables are then used to perform an
:ref:`alternating update<alternating_updates>` on the Lagrange multipliers.

Augmented Lagrangian coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recall that the update for the dual variables in an
:ref:`Augmented Lagrangian formulation<augmented_lagrangian_formulation>` is:

.. math::
    \lambda_{g, t+1} &= \left[\lambda_{g, t} + c_t g(x_{t+1}) \right]^+ \\
    \lambda_{h, t+1} &= \lambda_{h, t} + c_t h(x_{t+1})

This corresponds exactly to a (projected) gradient ascent update on the dual
variables with "step size" :math:`c_t` on the function:

.. math::
    \mathcal{L}_{c_t}(x_{t+1}, \lambda) \triangleq &  \, \, {\color{gray} \overbrace{ f(x_{t+1}) +\frac{c_t}{2} ||g(x_{t+1}) \odot \mathbf{1}_{g(x_{t+1}) \ge 0 \vee \lambda_{g} > 0}||^2 +  \frac{c_t}{2} ||h(x_{t+1})||^2}^{\text{do not contribute to gradient } \nabla_{\lambda} \mathcal{L}(x_{t+1}, \lambda)|_{\lambda = \lambda_t}}} \\
    &  \, \, +  \lambda_{g}^{\top} \, g(x_{t+1}) + \lambda_{h}^{\top} \, h(x_{t+1})

Therefore, the sequence of Augmented Lagrangian coefficients can be identified
with a :ref:`scheduler on the dual learning rate<dual_lr_scheduler>`.

.. math::
    \lambda_{t+1} = \texttt{dual_optimizer_update} \left( \lambda_{t}, {\color{red} \mathbf{-}} \nabla_{\lambda} \mathcal{L}({\color{red} x_{t+1}}, \lambda_t) , \texttt{lr} = c_t\right)


As in the :ref:`default parameter updates<basic_parameter_updates>`, we
explicitly include a negative sign in front of the dual gradient to denote the
`maximization` performed in the dual variables, since PyTorch optimizers use a
minimization convention. **Cooper** handles the gradient sign flip internally.



Example
~~~~~~~


.. code-block:: python
    :linenos:
    :emphasize-lines: 19, 20, 23, 31, 47, 58

    import torch
    import math
    import cooper

    class MyCustomCMP(cooper.ConstrainedMinimizationProblem):
        # ConstrainedMinimizationProblem with 3 inequality constraints and 4 equality constraints
        ...

    cmp = MyCustomCMP()

    params = torch.nn.Parameter(...)
    primal_optimizer = torch.optim.SGD([params], lr=1e-2)

    # Set the dual_optimizer base learning rate to 1.0 since the penalty coefficient
    # for the Augmented Lagrangian is controlled via the dual scheduler
    dual_optimizer = cooper.optim.partial_optimizer(torch.optim.SGD, lr=1.0)

    # Increasing Augmented Lagrangian coefficient schedule
    dual_scheduler = cooper.optim.partial_scheduler(
        torch.optim.lr_scheduler.LambdaLR, lr_lambda=lambda step: math.sqrt(step / 100)
    )

    formulation = cooper.formulation.AugmentedLagrangianFormulation(cmp)

    constrained_optimizer = cooper.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizers=primal_optimizer,
        dual_optimizer=dual_optimizer,
        dual_scheduler=dual_scheduler,
        alternation_type=AlternationType.PRIMAL_DUAL, # Remember that ALM performs alternating updates
    )

    # We need to manually trigger the creation of the Lagrange multipliers (formulation state)
    # and the link them to the dual optimizer.
    # This is so that the dual scheduler is fully initialized before we start training, since
    # we need it to compute the Augmented Lagrangian coefficient.
    formulation.create_state_from_metadata(
        dtype=params.dtype, device=device, ineq_size=torch.Size([3]), eq_size=torch.Size([4])
    )
    coop.instantiate_dual_optimizer_and_scheduler()

    for step_id in range(1000):
        coop.zero_grad()

        lagrangian = formulation.compute_lagrangian(
            aug_lag_coeff_scheduler=coop.dual_scheduler,
            closure=cmp.closure,
            params=params,
        )
        formulation.backward(lagrangian)

        # We need to pass the closure or defect_fn to perform the alternating updates
        # required by the Augmented Lagrangian method.
        coop.step(defect_fn=cmp.defect_fn, params=params) # Or: closure=cmp.closure

        # Remember that you need to call the dual_scheduler manually!
        coop.dual_scheduler.step()

--------------------------------------------------------------------------------

.. _multiple-primal_optimizers:

Multiple primal optimizers
--------------------------

When constructing a :py:class:`~cooper.optim.constrained_optimizers.ConstrainedOptimizer`,
one or multiple primal optimizers (grouped in a list) can be provided. Allowing
for multiple primal optimizers is useful when setting separate groups of primal
variables to have different optimizer classes and hyperparameters.

When a list of optimizers is provided for the ``primal_optimizers`` argument, they are
treated "as if they were a single optimizer". In particular, all primal optimizers
operations such as :py:meth:`optimizer.step()<torch.optim.Optimizer.step>` are
executed simultaneously (without intermediate calls to
:py:meth:`cmp.closure()<cooper.problem.ConstrainedMinimizationProblem.closure>` or
:py:meth:`formulation.backward(lagrangian)<cooper.formulation.Formulation.backward>`).
