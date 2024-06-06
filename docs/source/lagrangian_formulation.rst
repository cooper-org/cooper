.. _lagrangian_formulations:

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

A pair :math:`(x^*,\lambda^*)` is called a *saddle-point* of
:math:`\mathcal{L}(x,\lambda)` if for all :math:`(x,\lambda)`,

.. math::
    \mathcal{L}(x^*,\lambda) \le \mathcal{L}(x^*,\lambda^*) \le \mathcal{L}(x,\lambda^*).


This approach can be interpreted as a zero-sum two-player game, where the
"primal" player :math:`x` aims to minimize :math:`\mathcal{L}(x,\lambda)` and
the goal of the "dual" player :math:`\lambda` is to maximize
:math:`-\mathcal{L}(x,\lambda)` (or equiv. minimize
:math:`-\mathcal{L}(x,\lambda)`).

Note that the notion of a saddle-point of the Lagrangian is in fact equivalent
to that of a (pure) Nash equilibrium of the zero-sum game. If
:math:`(x^*,\lambda^*)` is a saddle-point of :math:`\mathcal{L}(x,\lambda)`,
then, by definition, neither of the two players can improve their payoffs by
unilaterally deviating from :math:`(x^*,\lambda^*)`.

In the context of a convex CMP (convex objectives, constraints and
:math:`\Omega`), given certain technical conditions (e.g. `Slater's condition
<https://en.wikipedia.org/wiki/Slater%27s_condition>`_
(see:math:`\S` 5.2.3 in :cite:p:`boyd2004convex`), or compactness of the domains),
the existence of a pure Nash equilibrium is guaranteed :cite:p:`vonNeumann1928theorie`.

.. warning::

    A constrained non-convex problem might have an optimal feasible solution,
    and yet its Lagrangian might not have a pure Nash equilibrium. See example
    in Fig 1. of :cite:t:`cotter2019JMLR`.

.. .. admonition:: Theorem (:math:`\S` 5.2.3, :cite:t:`boyd2004convex`)
..     :class: hint

..     Convex problem + Slater condition :math:`\Rightarrow` Strong duality

.. .. admonition:: Theorem (:math:`\S` 5.4.2, :cite:t:`boyd2004convex`)
..     :class: hint

..     (:math:`x^*, \lambda^*`) primal and dual optimal and strong duality
..     :math:`\Leftrightarrow` (:math:`x^*, \lambda^*`) is a saddle point of the
..     Lagrangian.

.. .. admonition:: Theorem
..     :class: hint

..     Every convex CMP with compact domain (for which strong duality holds) has
..     a Lagrangian for which a saddle point (i.e. pure Nash Equilibrium) exists.
..     (cite von Neumann?)



.. autoclass:: LagrangianFormulation
    :members:


Damped-Lagrangian Formulation
-----------------------------

The Damped-Lagrangian Formulation modifies the traditional Lagrangian approach to address oscillations that can occur in the optimization process when constraints are suddenly satisfied or violated.
This methodology, termed the *Modified Differential Method of Multipliers*, was initially proposed by John C. Platt and Alan H. Barr in 1988 :cite:`platt1987constrained`
and further explored in contemporary discussions on making machine learning algorithms more tunable, as highlighted in a recent blog post :cite:`engraved2024tunable`.

Overview
^^^^^^^^
The standard Lagrangian multipliers approach can lead to oscillatory behavior as the Lagrangian multiplier \(\lambda\) grows uncontrolled when constraints are breached.
Upon meeting the constraints, a high \(\lambda\) value can push the solution away from the constraint boundary, resulting in oscillations around the optimal solution.
The Damped-Lagrangian formulation introduces a damping mechanism to stabilize these oscillations and enhance convergence.

Theoretical Background
^^^^^^^^^^^^^^^^^^^^^^
The principal adjustment involves introducing a damping term to the update rule of \(\lambda\), analogous to damping in a physical oscillatory system,
to prevent excessive fluctuations and promote stability.

Advantages
^^^^^^^^^^
1. **Reduced Oscillations**: Introduces damping to minimize oscillatory behaviors, leading to more stable convergence.
2. **Flexibility**: Effective across both convex and concave Pareto fronts.
3. **Hyper-parameter Tuning**: The damping hyper-parameter allows for balancing quick convergence with adequate exploration of the solution space.

Considerations
^^^^^^^^^^^^^^
- **Hyper-parameter Selection**: Introduces a new hyper-parameter, the damping factor, which necessitates careful tuning to optimize convergence dynamics without altering the final solution.

.. autoclass:: DampedLagrangianFormulation
    :members:

Proxy-Lagrangian Formulation
----------------------------

.. autoclass:: ProxyLagrangianFormulation
    :members:

Base Lagrangian Formulation
----------------------------

.. autoclass:: BaseLagrangianFormulation
    :members:
