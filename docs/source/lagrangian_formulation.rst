.. currentmodule:: cooper.lagrangian_formulation

Lagrangian Formulations
=======================

Once equipped with a :py:class:`~cooper.problem.ConstrainedMinimizationProblem`,
several algorithmic approaches can be adopted for finding an approximation to
the solution of the constrained problem.

Recall that we consider constrained minimization problems (CMPs) expressed as:

.. math::
    \min_{x \in \Omega} & \,\, f(x) \\
    \text{s.t. } & \,\, g(x) \le \mathbf{0} \\
                 & \,\, h(x) = \mathbf{0}

Lagrangian Formulation
----------------------

The *Lagrangian* problem associated with the CMP above is given by:

.. math::
    \min_{x \in \Omega} \max_{\lambda_g \ge 0, \, \lambda_h} \mathcal{L}(x,\lambda) \triangleq f(x) + \lambda_g^{\top} g(x) + \lambda_h^{\top} h(x)

The vectors :math:`\lambda_g` and :math:`\lambda_h` are called the **Lagrange
multipliers** or **dual variables** associated with the CMP. Observe that
:math:`\mathcal{L}(x,\lambda)` is a concave function of :math:`\lambda` regardless
of the convexity properties of :math:`f, g` and :math:`h`.

This approach can be interpreted as a zero-sum two-player game, where the
"primal" player :math:`x` aims to minimize :math:`\mathcal{L}(x,\lambda)` and the
goal of the "dual" player :math:`\lambda` is to minimize :math:`-\mathcal{L}(x,\lambda)`.

.. admonition:: Theorem (:math:`\S` 5.2.3, :cite:t:`boyd2004convex`)

    Convex problem + Slater condition :math:`\Rightarrow` Strong duality

.. admonition:: Theorem (:math:`\S` 5.4.2, :cite:t:`boyd2004convex`)

    (:math:`x^*, \lambda^*`) primal and dual optimal and strong duality
    :math:`\Leftrightarrow` (:math:`x^*, \lambda^*`) is a saddle point of the
    Lagrangian.

.. admonition:: Theorem

    Every convex CMP with compact domain (for which strong duality holds) has
    a Lagrangian for which a saddle point (i.e. pure Nash Equilibrium) exists.
    (cite von Neumann?)

.. warning::

    On a constrained non-convex problem might have an optimal feasible solution,
    and yet its Lagrangian might not have a pure Nash equilibrium. See example
    in Fig 1. of :cite:p:`cotter19JMLR`

.. autoclass:: LagrangianFormulation
    :members:
    :special-members: __init__

Proxy-Lagrangian Formulation
----------------------------

.. autoclass:: ProxyLagrangianFormulation
    :members:
