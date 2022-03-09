Multipliers
===========

.. automodule:: cooper.multipliers

Constructing a DenseMultiplier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To construct a ``DenseMultiplier``, you have to provide it with an initial
value. The shape of this initial value must match the shape of the constraint
defect it corresponds to.

The positivity argument should be specified when initializing the multiplier.
If set to ``True``, it enforces multipliers to be positive. For multipliers
associated with inequality constraints, you must specify ``positive=True``.

.. note::

    Multipliers are designed to be mostly handled by
    :py:class:`~cooper.problem.Formulation`\s. This includes:

    - Their initialization in  in :py:class:`~cooper.problem.BaseLagrangianFormulation`'s :py:meth:`~cooper.problem.BaseLagrangianFormulation.create_state`.
    - Ensuring that their shape and device matches that of constraint defects given by the :py:class:`~cooper.problem.CMPState` of the ``ConstrainedMinimizationProblem`` they are solving.
    - Using them for computing Lagrangians in :py:class:`~cooper.problem.LagrangianFormulation`'s :py:meth:`~cooper.problem.LagrangianFormulation.composite_objective`.

.. autoclass:: DenseMultiplier
    :members:

.. autoclass:: BaseMultiplier
    :members:
