Multipliers
===========

.. automodule:: cooper.multipliers

.. note::

    Multipliers are designed to be handled internaly by
    :py:class:`~cooper.problem.Formulation`\s. This includes:

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

We lazily initialize ``DenseMultiplier``\s in the
:py:meth:`~cooper.lagrangian_formulation.BaseLagrangianFormulation.create_state`
method of :py:class:`~cooper.lagrangian_formulation.BaseLagrangianFormulation`\s
when the first ``CMPState`` of the ``ConstrainedMinimizationProblem`` is evaluated.
That way we can ensure that the ``DenseMultiplier``\s are initialized on the
correct device and that their shape matches that of the constraint defects.
If initial values for multipliers are not specified during the initialization
of the ``BaseLagrangianFormulation``, each is initialized to zero.


We specify ``positive=True`` when a multiplier corresponds to an inequality
constraint to enforce its value to be positive. After having performed a step
on the dual variables inside the
:py:meth:`~cooper.constrained_optimizer.ConstrainedOptimizer.step` method of
:py:class:`~cooper.constrained_optimizer.ConstrainedOptimizer`, inequality
multipliers which are negative are projected to zero.
For multipliers corresponding to equality constraints, we specify
``positive=False``. No projection is employed in this case.

.. autoclass:: DenseMultiplier
    :members:

.. autoclass:: BaseMultiplier
    :members:
