(lagrangian-formulations)=

```{eval-rst}
.. currentmodule:: cooper.formulation.lagrangian
```

# Lagrangian Formulations

Once equipped with a {py:class}`~cooper.problem.ConstrainedMinimizationProblem`,
several algorithmic approaches can be adopted for finding an approximation to
the solution of the constrained problem.

Recall that we consider constrained minimization problems (CMPs) expressed as:

$$
\min_{x \in \Omega} & \,\, f(x) \\
\text{s.t. } & \,\, g(x) \le \mathbf{0} \\
             & \,\, h(x) = \mathbf{0}
$$

## Lagrangian Formulation

The *Lagrangian* problem associated with the CMP above is given by:

$$
\min_{x \in \Omega} \max_{{\lambda^g} \ge 0, \, {\lambda^h}} \mathcal{L}(x,\lambda) \triangleq f(x) + {\lambda^g}^{\top} g(x) + {\lambda^h}^{\top} h(x)
$$

The vectors ${\lambda^g}$ and ${\lambda^h}$ are called the **Lagrange
multipliers** or **dual variables** associated with the CMP. Observe that
$\mathcal{L}(x,\lambda)$ is a concave function of $\lambda$ regardless
of the convexity properties of $f, g$ and $h$.

A pair $(x^*,\lambda^*)$ is called a *saddle-point* of
$\mathcal{L}(x,\lambda)$ if for all $(x,\lambda)$,

$$
\mathcal{L}(x^*,\lambda) \le \mathcal{L}(x^*,\lambda^*) \le \mathcal{L}(x,\lambda^*).
$$

This approach can be interpreted as a zero-sum two-player game, where the
"primal" player $x$ aims to minimize $\mathcal{L}(x,\lambda)$ and
the goal of the "dual" player $\lambda$ is to maximize
$\mathcal{L}(x,\lambda)$ (or equiv. minimize
$-\mathcal{L}(x,\lambda)$).

Note that the notion of a saddle-point of the Lagrangian is in fact equivalent
to that of a (pure) Nash equilibrium of the zero-sum game. If
$(x^*,\lambda^*)$ is a saddle-point of $\mathcal{L}(x,\lambda)$,
then, by definition, neither of the two players can improve their payoffs by
unilaterally deviating from $(x^*,\lambda^*)$.

In the context of a convex CMP (convex objectives, constraints and
$\Omega$), given certain technical conditions (e.g. [Slater's condition](https://en.wikipedia.org/wiki/Slater%27s_condition)
(see $\S$ 5.2.3 in {cite:p}`boyd2004convex`), or compactness of the domains),
the existence of a pure Nash equilibrium is guaranteed {cite:p}`vonNeumann1928theorie`.

:::{warning}
A constrained non-convex problem might have an optimal feasible solution,
and yet its Lagrangian might not have a pure Nash equilibrium. See example
in Fig 1. of {cite:t}`cotter2019JMLR`.
:::

% .. admonition:: Theorem (:math:`\S` 5.2.3, :cite:t:`boyd2004convex`)

% :class: hint

% Convex problem + Slater condition :math:`\Rightarrow` Strong duality

% .. admonition:: Theorem (:math:`\S` 5.4.2, :cite:t:`boyd2004convex`)

% :class: hint

% (:math:`x^*, \lambda^*`) primal and dual optimal and strong duality

% :math:`\Leftrightarrow` (:math:`x^*, \lambda^*`) is a saddle point of the

% Lagrangian.

% .. admonition:: Theorem

% :class: hint

% Every convex CMP with compact domain (for which strong duality holds) has

% a Lagrangian for which a saddle point (i.e. pure Nash Equilibrium) exists.

% (cite von Neumann?)

```{eval-rst}
.. autoclass:: LagrangianFormulation
    :members:

```

```{eval-rst}
.. currentmodule:: cooper.formulation.augmented_lagrangian
```

(augmented-lagrangian-formulation)=

## Augmented Lagrangian Formulation

The Augmented Lagrangian Method (ALM) considers a `sequence` of unconstrained
minimization problems on the primal variables:

$$
L_{c_t}(x, \lambda^t) \triangleq f(x) + \lambda_{g, t}^{\top} \, g(x) + \lambda_{h, t}^{\top} \, h(x) + \frac{c_t}{2} ||g(x) \odot \mathbf{1}_{g(x_t) \ge 0 \vee \lambda_{g, t} > 0}||^2 +  \frac{c_t}{2} ||h(x_t)||^2
$$

This problem is (approximately) minimized over the primal variables to obtain:

$$
x^{t+1} = \arg \min_{x \in \Omega} \mathcal{L}_{c^t}(x, \lambda^t)
$$

The found $x^{t+1}$ is used to update the estimate for the Lagrange multiplier:

$$
\begin{align}     \lambda_{t+1}^g &= \left[\lambda_t^g + c^t g(x^{t+1}) \right]^+ \\     \lambda_{t+1}^h &= \lambda_t^h + c^t h(x^{t+1}) \end{align}
$$

The main advantage of the ALM compared to the quadratic penalty method
(see $\S$ 4.2.1 in {cite:p}`bertsekas1999NonlinearProgramming`) is that
(under some reasonable assumptions), the algorithm can be successful without
requiring the unbounded increase of the penalty parameter sequence $c^t$.
The use of explicit estimates for the Lagrange multipliers contribute to
avoiding the  ill-conditioning that is inherent in the quadratic penalty method.

See $\S$ 4.2.1 in {cite:p}`bertsekas1999NonlinearProgramming` and
$\S$ 17 in {cite:p}`nocedal2006NumericalOptimization` for a comprehensive
treatment of the Augmented Lagrangian method.

:::{important}
Please visit {ref}`this section<augmented_lagrangian_const_opt>` for
practical considerations on using the Augmented Lagrangian method in
**Cooper**.

In particular, the sequence of penalty coefficients $c_t$ is handled
in **Cooper** as a
{ref}`scheduler on the dual learning rate<dual_lr_scheduler>`.
:::

```{eval-rst}
.. autoclass:: AugmentedLagrangianFormulation
    :members:


```

```{eval-rst}
.. currentmodule:: cooper.formulation.lagrangian
```

## Proxy-Lagrangian Formulation

```{eval-rst}
.. autoclass:: ProxyLagrangianFormulation
    :members:
```

## Base Lagrangian Formulation

```{eval-rst}
.. autoclass:: BaseLagrangianFormulation
    :members:
```
