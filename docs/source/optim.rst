Optim Module
============

.. currentmodule:: cooper.optim

.. _partial_optimizer_instantiation:

Partial optimizer instantiation
-------------------------------

When constructing a :py:class:`~cooper.optim.constrained_optimizesr.ConstrainedOptimizer`, the
``dual_optimizer`` parameter is expected to be a
:py:class:`torch.optim.Optimizer` for which the ``params`` argument has **not
yet** been passed. The rest of the instantiation of the ``dual_optimizer`` is
handled internally by **Cooper**.

The :py:meth:`cooper.optim.partial_optimizer` method below allows you to provide a
configuration for your ``dual_optimizer``\'s hyperparameters (e.g. learning
rate, momentum, etc.)

.. automethod:: cooper.optim.partial_optimizer

Learning rate schedulers
------------------------

**Cooper** supports learning rate schedulers for the primal and dual optimizers.
Recall that **Cooper** handles the primal and dual optimizers in slightly
different ways: the primal optimizer is "fully" instantiated by the user, while
we expect a "partially" instantiated dual optimizer. We follow a similar pattern
for the learning rate schedulers.

**Example:**

    .. code-block:: python
        :linenos:
        :emphasize-lines: 7,8,10,15

        from torch.optim.lr_scheduler import StepLR, ExponentialLR

        ...
        primal_optimizer = torch.optim.SGD(...)
        dual_optimizer = cooper.optim.partial_optimizer(...)

        primal_scheduler = StepLR(primal_optimizer, step_size=1, gamma=0.1)
        dual_scheduler = cooper.optim.partial_scheduler(ExponentialLR, **scheduler_kwargs)

        const_optim = cooper.ConstrainedOptimizer(..., primal_optimizer, dual_optimizer, dual_scheduler)

        for step in range(num_steps):
            ...
            const_optim.step() # Cooper calls dual_scheduler.step() internally
            primal_scheduler.step()  # You must call this explicitly

Primal learning rate scheduler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _primal_lr_scheduler:

You must instantiate the scheduler for the learning rate used by each
``primal_optimizer`` and call the scheduler's ``step`` method explicitly, as is
usual in Pytorch. See :py:mod:`torch.optim.lr_scheduler` for details.

Dual learning rate scheduler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _dual_lr_scheduler:

When constructing a
:py:class:`~cooper.optim.constrained_optimizers.ConstrainedOptimizer`,
the ``dual_scheduler`` parameter is expected to be a *partially instantiated*
learning rate scheduler from Pytorch, for which the ``optimizer`` argument has
**not yet** been passed. The :py:meth:`cooper.optim.partial_scheduler` method
allows you to provide a  configuration for your ``dual_scheduler``\'s
hyperparameters. The rest of the instantiation of the ``dual_scheduler`` is
managed internally by **Cooper**.

.. note::

    The call to the ``step()`` method of the dual optimizer is handled
    internally by **Cooper**. However, you must perform the call to the dual
    scheduler's ``step`` method manually. This will usually come after several
    calls to :py:meth:`cooper.optim.constrained_optimizers.ConstrainedOptimizer.step`.

    The reasoning behind this design is to provide you, the user, with greater
    visibility and control on the dual learning rate scheduler. For example, you
    might want to synchronize the changes in the dual learning rate scheduler
    depending on the number of training epochs ellapsed so far.

    This flexibility is also desirable when using an
    :ref:`Augmented Lagrangian Formulation<augmented_lagrangian_formulation>`,
    since the penalty coefficient for the augmented Lagrangian can be controlled
    directly via the dual learning rate scheduler.


``PartialScheduler`` Class
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: cooper.optim.partial_scheduler


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
    :py:class:`~cooper.optim.constrained_optimizers.ConstrainedOptimizer`, the primal
    and dual optimizers must **both** be instances of classes inheriting from
    :py:class:`ExtragradientOptimizer`.

    When provided with extrapolation-capable optimizers, **Cooper** will
    automatically trigger the calls to the extrapolation function.

    Due to the calculation of gradients at the "look-ahead" point
    :math:`\omega_{t+1/2}`, the call to
    :py:meth:`cooper.optim.constrained_optimizers.ConstrainedOptimizer.step` requires
    passing the parameters needed for the computation of the
    :py:meth:`cooper.problem.ConstrainedMinimizationProblem.closure`.


    **Example:**

    .. code-block:: python
        :linenos:
        :emphasize-lines: 11,12,31

        model = ...

        cmp = cooper.ConstrainedMinimizationProblem()
        formulation = cooper.Formulation(...)

        # Non-extra-gradient optimizers
        primal_optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        dual_optimizer = cooper.optim.partial_optimizer(torch.optim.SGD, lr=1e-3)

        # Extra-gradient optimizers
        primal_optimizer = cooper.optim.ExtraSGD(model.parameters(), lr=1e-2)
        dual_optimizer = cooper.optim.partial_optimizer(cooper.optim.ExtraSGD, lr=1e-3)

        const_optim = cooper.ConstrainedOptimizer(
            formulation=formulation,
            primal_optimizers=primal_optimizer,
            dual_optimizer=dual_optimizer,
        )

        for step in range(num_steps):
            const_optim.zero_grad()
            lagrangian = formulation.compute_lagrangian(cmp.closure, model, inputs)
            formulation.backward(lagrangian)

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
