Optim
=====

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

The :py:meth:`cooper.optim.partial_optimizer` method below allows you to provide a
configuration for your ``dual_optimizer``\'s hyperparameters (e.g. learning
rate, momentum, etc.)

.. automethod:: cooper.optim.partial_optimizer


.. _extra-gradient_optimizers:

Extra-gradient optimizers
-------------------------

The extra-gradient method :cite:p:`korpelevich1976extragradient` is a standard
approach for solving min-max games as those appearing in the
:py:class:`~cooper.formulation.LagrangianFormulation`.


Given a Lagrangian :math:`\mathcal{L}(x,\lambda)`, define the joint variable
:math:`\omega = (x,\lambda)` and the "gradient" operator:

.. math::

    F(\omega) = [\nabla_x \mathcal{L}(x,\lambda), -\nabla_{\lambda} \mathcal{L}(x,\lambda)]^{\top}

The extra-gradient update can be summarized as:

.. math::

    \omega_{t+1/2} &= P_{\Omega}[\omega_{t+} - \eta F(\omega_{t})] \\
    \omega_{t+1} &= P_{\Omega}[\omega_{t} - \eta F(\omega_{t+1/2})]

.. note::

    In the *unconstrained* case, the extra-gradient update is "intrinsically
    different" from that of Nesterov momentum :cite:p:`gidel2018variational`.
    The current version of **Cooper** raises a :py:class:`RuntimeError` when
    trying to use an :py:class:`ExtragradientOptimizer`. This
    restriction might be lifted in future releases.

The implementations of :py:class:`~cooper.optim.ExtraSGD` and
:py:class:`~cooper.optim.ExtraAdam` included in **Cooper** are minor edits from
those originally written by `Hugo Berard
<https://github.com/GauthierGidel/Variational-Inequality-GAN/blob/master/optim/extragradient.py>`_\.
:cite:t:`gidel2018variational` provides a concise presentation of the
extra-gradient in the context of solving Variational Inequality Problems.

.. warning::

    If you decide to use extra-gradient optimizers for defining a
    :py:class:`~cooper.constrained_optimizer.ConstrainedOptimizer`, the primal
    and dual optimizers must **both** be instances of classes inheriting from
    :py:class:`ExtragradientOptimizer`.

    When provided with extrapolation-capable optimizers, **Cooper** will
    automatically trigger the calls to the extrapolation function.

    Due to the calculation of gradients at the "look-ahead" point
    :math:`\omega_{t+1/2}`, the call to
    :py:meth:`cooper.constrained_optimizer.ConstrainedOptimizer.step` requires
    passing the parameters needed for the computation of the
    :py:meth:`cooper.problem.ConstrainedMinimizationProblem.closure`.


    **Example:**

    .. code-block:: python
        :linenos:
        :emphasize-lines: 11,12,31

        model = ...

        cmp = cooper.ConstrainedMinimizationProblem(is_constrained=True)
        formulation = cooper.problem.Formulation(...)

        # Non-extra-gradient optimizers
        primal_optimizer = cooper.optim.SGD(model.parameters(), lr=1e-2)
        dual_optimizer = cooper.optim.partial_optimizer(cooper.optim.SGD, lr=1e-3)

        # Extra-gradient optimizers
        primal_optimizer = cooper.optim.ExtraSGD(model.parameters(), lr=1e-2)
        dual_optimizer = cooper.optim.partial_optimizer(cooper.optim.ExtraSGD, lr=1e-3)

        const_optim = cooper.ConstrainedOptimizer(
            formulation=formulation,
            primal_optimizer=primal_optimizer,
            dual_optimizer=dual_optimizer,
        )

        for step in range(num_steps):
            const_optim.zero_grad()
            lagrangian = formulation.composite_objective(cmp.closure, model, inputs)
            formulation.custom_backward(lagrangian)

            # Non-extra-gradient optimizers
            # Passing (cmp.closure, model, inputs) to step will simply be ignored
            const_optim.step()

            # Extra-gradient optimizers
            # Must pass (cmp.closure, model, inputs) to step
            const_optim.step(cmp.closure, model, inputs)


.. autoclass:: ExtragradientOptimizer
    :members:

.. autoclass:: ExtraSGD
    :members:

.. autoclass:: ExtraAdam
    :members:

Learning Rate Schedulers
------------------------

**Cooper** supports learning rate schedulers for the primal and dual optimizers.

Primal lr scheduler
~~~~~~~~~~~~~~~~~~~

.. _primal_lr_scheduler:

You must instantiate the scheduler for the learning rate of a
``primal_optimizer`` and use it as in standard Pytorch. See
:py:mod:`torch.optim` for details.

**Example:**

    .. code-block:: python
        :linenos:
        :emphasize-lines: 8,13

        from torch.optim.lr_scheduler import StepLR

        ...
        primal_optimizer = cooper.optim.SGD(...)
        dual_optimizer = cooper.optim.partial_optimizer(...)
        const_optim = cooper.ConstrainedOptimizer(...)

        primal_scheduler = StepLR(primal_optimizer, step_size=1, gamma=0.1)

        for step in range(num_steps):
            ...
            const_optim.step()
            primal_scheduler.step()

Dual lr scheduler
~~~~~~~~~~~~~~~~~

.. _dual_lr_scheduler:

When constructing a :py:class:`~cooper.constrained_optimizer.ConstrainedOptimizer`,
the ``dual_scheduler`` parameter is expected to be a partially instantiated
Pytorch learning rate scheduler, for which the ``optimizer`` argument has **not
yet** been passed. The rest of the instantiation of the ``dual_scheduler`` is
handled internally by **Cooper**.

The :py:meth:`cooper.optim.partial_scheduler` method below allows you to provide
a configuration for your ``dual_scheduler``\'s hyperparameters.

.. automethod:: cooper.optim.partial_scheduler