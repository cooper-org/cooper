.. currentmodule:: cooper.problem

Constrained Minimization Problem
================================

We consider constrained minimization problems (CMPs) expressed as:

.. math::
    \min_{x \in \Omega} & \,\, f(x) \\
    \text{s.t. } & \,\, g(x) \le \mathbf{0} \\
                 & \,\, h(x) = \mathbf{0}

Here :math:`\Omega` represents the domain of definition of the functions
:math:`f, g` and :math:`h`. Note that :math:`f` is a scalar-valued function.
We group together all the inequality and equality constraints into the
(potentially) vector-valued mappings :math:`g` and :math:`h`. In other words,
a component function :math:`h_i(x)` is associated with the scalar constraint
:math:`h_i(x) \le 0`.

.. admonition:: Brief notes on conventions and terminology

    * We refer to :math:`f` as the **loss** or **main objective** to be minimized.
    * Many authors prefer making the constraint levels explicit (e.g.
      :math:`g(x) \le \mathbf{\epsilon}`). To improve the readability of the
      code, we adopt the convention that the constraint levels have been
      "absorbed" in the definition of the functions :math:`g` and :math:`h`.
    * Based on this convention, we use the terms **defect** and **constraint violation**
      interchangeably to denote the quantities :math:`g(x)` or :math:`h(x)`. Note
      that equality constraints :math:`h(x)` are satisfied *only* when their
      defect is zero. On the other hand, a *negative* defect for an inequality
      constraint means that the constraint is *strictly* satisfied; while a
      *positive* defect means that the inequality constraint is being violated.

Constrained Minimization Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We say that a CMP is a convex problem if :math:`f, g` and :math:`h` are all convex
functions of :math:`x` and :math:`\Omega` is convex.

.. warning::
    **Cooper** is primarily oriented towards **non-convex** CMPs that arise
    in many machine/deep learning settings. Whenever possible, we provide references
    to appropriate literature describing convergence results for our implemented
    (under suitable assumptions). In general, however, the use of Lagrangian-based
    approached for solving non-convex CMPs does not come with guarantees regarding
    optimality or feasibility.

    If you are dealing with optimization problems under "nicely behaved" convex
    constraints (e.g. :math:`L_p`-balls, cones) we encourage you to check out
    `CHOP <https://github.com/openopt/chop>`_. If your problems involves "manifold"
    constraints (e.g. orthogonal or PSD matrices), you might consider using
    `GeoTorch <https://github.com/Lezcano/geotorch>`_.

.. autoclass:: ConstrainedMinimizationProblem
    :members:

CMP State
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CMPState
    :members:

Formulation
~~~~~~~~~~~

Formulations denote mathematical or algorithmic techniques aimed at solving a
specific (family of) CMP. **Cooper** is heavily, but not exclusively, oriented
towards Lagrangian-based formulations. You can find more details in
:doc:`lagrangian_formulation`.

.. autoclass:: Formulation
    :members:
