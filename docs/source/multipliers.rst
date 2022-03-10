Multipliers
===========

.. automodule:: cooper.multipliers

.. note::

    Multipliers are mostly handled internally by the
    :py:class:`~cooper.problem.Formulation`\s. This handling includes:

    - Their initialization in the
      :py:meth:`~cooper.lagrangian_formulation.BaseLagrangianFormulation.create_state`
      method of :py:class:`~cooper.lagrangian_formulation.BaseLagrangianFormulation`.
    - Ensuring that their shape and device matches that of the constraint
      defects provided by the :py:class:`~cooper.problem.CMPState` of the
      considered :py:class:`~cooper.problem.ConstrainedMinimizationProblem`.
    - Using them for computing Lagrangians in the
      :py:meth:`~cooper.lagrangian_formulation.LagrangianFormulation.composite_objective`
      method of :py:class:`~cooper.lagrangian_formulation.LagrangianFormulation`.

Constructing a DenseMultiplier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To construct a ``DenseMultiplier``, its initial value ``init`` must be provided.
The shape of this initial value must match the shape of the constraint defect it
corresponds to.

We lazily initialize ``DenseMultiplier``\s via a call to the
:py:meth:`~cooper.lagrangian_formulation.BaseLagrangianFormulation.create_state`
method of a :py:class:`~cooper.lagrangian_formulation.BaseLagrangianFormulation`
object. This happens when the first ``CMPState`` of the
``ConstrainedMinimizationProblem`` is evaluated. That way we can ensure that the
``DenseMultiplier``\s are initialized on the correct device and that their shape
matches that of the constraint defects. If initial values for multipliers are
not specified during the initialization of the ``BaseLagrangianFormulation``,
they are initialized to zero.

After having performed a step on the dual variables inside the
:py:meth:`~cooper.constrained_optimizer.ConstrainedOptimizer.step` method of
:py:class:`~cooper.constrained_optimizer.ConstrainedOptimizer`, a call to
:py:meth:`~DenseMultiplier.project_` ensures that the multiplier values are
admissible. It is possible to override the :py:meth:`~DenseMultiplier.project_`
method in order to apply a custom projection to the multipliers.

In the case of a :py:class:`~cooper.lagrangian_formulation.LagrangianFormulation`,
the projection only affects the Lagrange multipliers associated with the inequality
constraints.

The flag ``positive`` denotes whether a multiplier corresponds to an inequality
or equality constraint, and thus whether the multiplier value must be
lower-bounded by zero or not. ``positive=True`` corresponds to inequality
constraints, while ``positive=False`` corresponds to equality constraints.

.. autoclass:: DenseMultiplier
    :members:

Extension to other Multipliers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Certain optimization problems comprise a very large number of constraints. For
example, in a learning task one might impose a constraint *per data-point*.
In these settings, explicitly maintaining one Lagrange multiplier per constraint
becomes impractical :cite:p:`narasimhan2019multiplier`. We provide a
``BaseMultiplier`` class which can be easily extended to accommodate situations
which employ sparse multipliers or even a model that predicts the value of the
multiplier based on some properties or "features" of each constraint.

.. autoclass:: BaseMultiplier
    :members:
