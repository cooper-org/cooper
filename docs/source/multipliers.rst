.. _multipliers:

Multipliers
===========

.. currentmodule:: cooper.multipliers

.. note::

    Multipliers are mostly handled internally by the
    :py:class:`~cooper.formulation.Formulation`\s. This handling includes:

    - Their initialization in the
      :py:meth:`~cooper.formulation.lagrangian.BaseLagrangianFormulation.create_state`
      method of :py:class:`~cooper.formulation.lagrangian.BaseLagrangianFormulation`.
    - Ensuring that their shape and device matches that of the constraint
      defects provided by the :py:class:`~cooper.problem.CMPState` of the
      considered :py:class:`~cooper.problem.ConstrainedMinimizationProblem`.
    - Using them for computing Lagrangians in the
      :py:meth:`~cooper.formulation.lagrangian.LagrangianFormulation.compute_lagrangian`
      method of :py:class:`~cooper.formulation.lagrangian.LagrangianFormulation`.

Constructing a DenseMultiplier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The construction of a ``DenseMultiplier`` requires its initial value ``init`` to
be provided. The shape of ``init`` should match that shape of the corresponding
constraint defect. Recall that in :ref:`lagrangian_formulations`, the
calculation of the Lagrangian involves an inner product between the multipliers
and the constraint violations.

We lazily initialize ``DenseMultiplier``\s via a call to the
:py:meth:`~cooper.formulation.lagrangian.BaseLagrangianFormulation.create_state`
method of a :py:class:`~cooper.formulation.lagrangian.BaseLagrangianFormulation`
object. This happens when the first ``CMPState`` of the
``ConstrainedMinimizationProblem`` is evaluated. That way we can ensure that the
``DenseMultiplier``\s are initialized on the correct device and that their shape
matches that of the constraint defects. If initial values for multipliers are
not specified during the initialization of the ``BaseLagrangianFormulation``,
they are initialized to zero.

After having performed a step on the dual variables inside the
:py:meth:`~cooper.optim.constrained_optimizers.ConstrainedOptimizer.step` method of
:py:class:`~cooper.optim.constrained_optimizers.ConstrainedOptimizer`, a call to
:py:meth:`~DenseMultiplier.project_` ensures that the multiplier values are
admissible. It is possible to override the :py:meth:`~DenseMultiplier.project_`
method in order to apply a custom projection to the multipliers.

In the case of a :py:class:`~cooper.formulation.lagrangian.LagrangianFormulation`,
the projection only affects the Lagrange multipliers associated with the inequality
constraints.

The flag ``positive`` denotes whether a multiplier corresponds to an inequality
or equality constraint, and thus whether the multiplier value must be
lower-bounded by zero or not. ``positive=True`` corresponds to inequality
constraints, while ``positive=False`` corresponds to equality constraints.

.. autoclass:: DenseMultiplier
    :members:

Extensions
^^^^^^^^^^

Certain optimization problems involve a very large number of constraints. For
example, in a learning task one might impose a constraint *per data-point*.
In these settings, explicitly maintaining one Lagrange multiplier per constraint
becomes impractical :cite:p:`narasimhan2020multiplier`. We provide a
``BaseMultiplier`` class which can be easily extended to accommodate situations
which employ sparse multipliers or even a model that predicts the value of the
multiplier based on some properties or "features" of each constraint.

.. autoclass:: BaseMultiplier
    :members:
