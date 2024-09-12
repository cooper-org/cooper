(overview)=

# Overview of Constrained Optimization

## Constrained Minimization Problems

We consider constrained minimization problems (CMPs) expressed as:

$$
\min_{\vx \in \reals^d} & \,\, f(\vx) \\ \text{s.t. }
& \,\, \vg(\vx) \le \vzero \\ & \,\, \vh(\vx) = \vzero
$$

:::{admonition} Conventions and terminology

- We refer to $f$ as the **loss** or **objective** to be minimized.
- We adopt the convention $\vg(\vx) \le 0$ for **inequality constraints** and $\vh(\vx) = 0$ for **equality constraints**. If your constraints are different, for example $\vg(\vx) \ge \epsilon$, you should provide **Cooper** with $\epsilon - \vg(\vx) \le 0$.
:::

:::{warning}
We use the term **constraint violation** to refer to $\vg(\vx)$ and $\vh(\vx)$. Equality constraints $h(x)$ are satisfied *only* when their violation is zero. On the other hand, a *negative* violation for an inequality constraint $g(x)$ means that the constraint is *strictly* satisfied; while a *positive* violation means that the inequality constraint is being violated.

Note that we call g(x) a violation even when g(x) < 0, even though the constraint is satisfied. This is different from some books that call |h(x)| a violation and relu(g(x)) a violation (this quantifies how much the constraint is violated).

We call g and h violations even when the constraints are satisfied.

For example, a negative violation for an inequality constraint $g(x)$ means that the constraint is strictly satisfied; while a positive violation means that the inequality constraint is being violated.

:::


We group together all the inequality constraints in $\vg$, and all the equality constraints in $\vh$.
In other words, $f$ is a scalar-valued function, whereas $\vg$ and $\vh$ are vector-valued functions with parameters $\vx$.
A component function $h_i(\vx)$ corresponds to the scalar constraint
$h_i(\vx) = 0$.


## Lagrangian problem

An approach for solving general non-convex CMPs is to formulate their Lagrangian and find a min-max point:

$$
\xstar, \lambdastar, \mustar = \argmin{\vx \in \reals^d} \, \, \argmax{\vlambda \ge \vzero, \vmu} \, \, \mathcal{L}(\vx, \vlambda, \vmu)
$$

where $\mathcal{L}(\vx, \vlambda, \vmu) = f(\vx) + \vlambda^\top \vg(\vx) + \vmu^\top \vh(\vx)$, and $\vlambda$ and $\vmu$ are the Lagrange multipliers associated with the inequality and equality constraints, respectively.
We refer to $\vx$ as the **primal variables** of the CMP, and $\vlambda$ and $\vmu$ as the **dual variables**.

An argmin-argmax point of the Lagrangian corresponds to a solution of the original CMP {cite:p}`boyd2004convex`. We refer to finding such a point as the **Lagrangian approach** to solving a constrained minimization problem. **Cooper** is primarily designed to solve CMPs using the Lagrangian approach, but it also implements alternative formulations such as the {py:class}`~cooper.formulation.AugmentedLagrangianFormulation` (see {doc}`formulations`).


## Min-max optimization

An approach for finding min-max points of the Lagrangian is doing gradient descent on the primal variables and gradient ascent on the dual variables. **Cooper** leverages PyTorch's automatic differentiation framework to do this efficiently. **Cooper** also implements variants of gradient descent-ascent such as the {py:class}`~cooper.optim.Extragradient` {cite:p}`korpelevich1976extragradient` (see {doc}`optimizers`).

:::{warning}
**Cooper** is primarily oriented towards **non-convex** CMPs that arise
in many deep learning applications. That is, problems for which one of
the functions $f, \vg, \vh$ is non-convex. While the techniques
implemented in **Cooper** are applicable to convex problems as well, we
recommend using specialized solvers for convex optimization problems whenever
possible.
:::


## Proxy Constraints

TODO(juan43ramirez)
