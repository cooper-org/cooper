(overview)=

# Overview of Constrained Optimization

## Constrained Minimization Problems

We consider constrained optimization problems expressed as:

$$
\min_{\vx \in \reals^d} & \,\, f(\vx) \\ \text{s.t. }
& \,\, \vg(\vx) \le \vzero \\ & \,\, \vh(\vx) = \vzero
$$

:::{admonition} Conventions and terminology

- We refer to $f$ as the **loss** or **objective** to be minimized.
- We adopt the convention $\vg(\vx) \le \vzero$ for **inequality constraints** and $\vh(\vx) = \vzero$ for **equality constraints**. If your constraints are different, for example $\vc(\vx) \ge \epsilon$, you should provide **Cooper** with $\vg(\vx) = \epsilon - \vc(\vx) \le \vzero$.
:::

:::{warning}
We use the term constraint violation to refer to both $\vg(\vx)$ and $\vh(\vx)$. For equality constraints, $\vh(\vx)$ is satisfied only when its violation is zero, i.e., $\vh(\vx) = \vzero$. For inequality constraints, a negative violation of $\vg(\vx)$ indicates the constraint is strictly satisfied (i.e., $\vg(\vx) < \vzero$), whereas a positive violation indicates the constraint is violated (i.e., $\vg(\vx) > \vzero$).

Note that we still refer to $\vg(\vx)$ and $\vh(\vx)$ as "violations" even when the constraint are satisfied. This differs from the convention in some of the optimization literature, which uses the term "violation" to refer to the amount by which a constraint is violated (i.e., $\max\{\vzero, \vg(\vx)\}$ for inequality constraints and $|\vh(\vx)|$ for equality constraints).
:::


We group together all the inequality constraints in $\vg$, and all the equality constraints in $\vh$.
In other words, $f$ is a scalar-valued function, whereas $\vg$ and $\vh$ are vector-valued functions with parameters $\vx$.
A component function $h_i(\vx)$ corresponds to the scalar constraint
$h_i(\vx) = 0$.


## The Lagrangian Approach

An approach for solving general nonconvex constrained optimization problems is to formulate their Lagrangian and find a min-max point:

$$
\xstar, \lambdastar, \mustar = \argmin{\vx \in \reals^d} \, \, \argmax{\vlambda \ge \vzero, \vmu} \, \, \Lag(\vx, \vlambda, \vmu)
$$

where $\Lag(\vx, \vlambda, \vmu) = f(\vx) + \vlambda^\top \vg(\vx) + \vmu^\top \vh(\vx)$ is the Lagrangian function associated with the constrained minimization problem. $\vlambda \geq \vzero$ and $\vmu$ are the Lagrange multipliers associated with the inequality and equality constraints, respectively.
We refer to $\vx$ as the **primal variables** of the CMP, and $\vlambda$ and $\vmu$ as the **dual variables**.

:::{note}
$\Lag(\vx,\vlambda, \vmu)$ is concave in $\vlambda$ and $\vmu$ regardless of the convexity properties of $f$, $\vg$, and $\vh$.
:::

An argmin-argmax point of the Lagrangian corresponds to a solution of the original CMP {cite:p}`boyd2004convex`. We refer to finding such a point as the **Lagrangian approach** to solving a constrained minimization problem. **Cooper** is primarily designed to solve constrained optimization problems using the Lagrangian approach, and it also implements alternative formulations such as the {py:class}`~cooper.formulation.AugmentedLagrangian` (see {doc}`formulations`).

:::{admonition} Why does **Cooper** use the Lagrangian approach?
**Cooper** is designed for solving constrained optimization problems that arise in deep learning applications. These problems are often **nonconvex** and **high-dimensional**, and may require **estimating constraints stochastically** from mini-batches of data. The Lagrangian approach is well-suited to these problems for several reasons:
- **Nonconvexity**. The Lagrangian approach does not require the loss or constraints to be convex or follow a specific structure, making it applicable to general nonconvex problems.
>
- **Scalability**. First-order optimization methods, such as gradient descent-ascent, can be used to find min-max points of the Lagrangian. These methods are well-supported by automatic differentiation frameworks such as PyTorch and scale to high-dimensional problems.
\
Moreover, the overhead (relative to unconstrained minimization) of storing and updating the Lagrange multipliers is generally negligible in deep learning problems, where the computational cost of calculating the loss, constraints, and their gradients represents the main bottleneck.
>
- **Stochastic estimates of the constraints**. Gradient-based methods can utilize stochastic estimates of the loss and constraints, making them applicatble to problems where computing the exact loss and constraints is prohibitively expensive.
:::

:::{warning}
**Cooper** is primarily designed for **nonconvex** constrained optimization problems that arise in many deep learning applications. While the techniques implemented in **Cooper** are applicable to convex problems as well, we recommend using specialized solvers for convex optimization problems whenever possible.
:::

## Min-max Optimization

A simple approach for finding min-max points of the Lagrangian is doing gradient descent on the primal variables and gradient ascent on the dual variables. Simultaneous **gradient descent-ascent** has the following updates:

$$
\vx_{t+1} &= \vx_t - \eta_{\vx} \nabla_{\vx} \Lag(\vx_t, \vlambda_t, \vmu_t) \\
\vlambda_{t+1} &= \left [ \vlambda_t + \eta_{\vlambda} \nabla_{\vlambda} \Lag(\vx_t, \vlambda_t, \vmu_t) \right ]_+ \\
\vmu_{t+1} &= \vmu_t + \eta_{\vmu} \nabla_{\vmu} \Lag(\vx_t, \vlambda_t, \vmu_t)
$$

where $\eta_{\vx}, \eta_{\vlambda}, \eta_{\vmu}$ are the step sizes for the primal and dual variables. The projection operator $[\cdot]_+$ ensures that the dual variables associated with the inequality constraints remain non-negative.

Plugging in the gradients of the Lagrangian, we get the following updates:

$$
\vx_{t+1} &= \vx_t - \eta_{\vx} \left [ \nabla_{\vx} f(\vx_t) + \vlambda_t^\top \nabla_{\vx} \vg(\vx_t) + \vmu_t^\top \nabla_{\vx} \vh(\vx_t) \right ] \\
\vlambda_{t+1} &= \left [ \vlambda_t + \eta_{\vlambda} \vg(\vx_t) \right ]_+ \\
\vmu_{t+1} &= \vmu_t + \eta_{\vmu} \vh(\vx_t)
$$

The primal updates follow a linear combination of the gradients of the loss and constraints, with the coefficients corresponding to the Lagrange multipliers. Larger values of a Lagrange multiplier result in a stronger influence of the corresponding constraint on the primal updates, promoting feasibility. Conversely, smaller values (or zero) reduce the influence of the constraint, prioritizing loss reduction.

On the other hand, the dual updates accumulate the constraint violations. Together with the primal updates, these ensure that the constraints are satisfied:
- **Inequality constraints**: When a constraint is violated ($\vg(\vx) > \vzero$), the corresponding Lagrange multiplier increases to penalize the violation. If the constraint is strictly satisfied ($\vg(\vx) < \vzero$), the multiplier decreases, allowing the focus to shift toward loss reduction.
- **Equality constraints**: For a positive (resp. negative) violation, the Lagrange multiplier increases (resp. decreases). The multiplier stabilizes when the constraint is satisfied ($\vh(\vx) = \vzero$).

**Cooper** leverages PyTorch's automatic differentiation framework to efficiently perform gradient-based optimization of the Lagrangian.
**Cooper** supports simultaneous gradient descent-ascent, as well as other variants like alternating gradient descent-ascent and the {py:class}`~cooper.optim.Extragradient` method {cite:p}`korpelevich1976extragradient`.

With **Cooper**, you can specify {py:class}`~torch.optim.Optimizer` objects for the primal and dual updates (see {doc}`optim`), allowing you to apply familiar optimization techniques such as Adam, just as you would when training deep neural networks.

## Optimality Conditions

TODO

Main question: when do I stop?

NOTE: subsection on approximately-stationary point of the Lagrangian for non-convex problems

Looking for a stationary point of the Lagrangian
Primal and dual conditions


These are equivalent to the **Karush-Kuhn-Tucker (KKT) conditions** for the original constrained minimization problem {cite:p}`boyd2004convex`.

:::{admonition} Terminating Conditions
:class: tip


The primal updates do not exclusively follow the gradient of the loss. Therefore, it is common for the loss to *increase* during the initial phase of the optimization, as this is often required to satisfy the constraints. After becoming feasible, however, the loss is expected to decrease until an equilibrium is reached.

The presented algorithm looks for a stationary point of the Lagrangian. Therefore, to ensure convergence, you should check that the gradient of the Lagrangian with respect to the primal variables $\vx$ is close to zero, and that complimentary slackness holds for the dual variables: $\vlambda \circ \vg = \vzero$ and $\vmu \circ \vh = \vzero$.
:::


(proxy)=
## Non-differentiable Constraints

{cite:t}`cotter2019proxy` introduce the concept of **proxy constraints** to address problems with non-differentiable constraints. In these cases, the gradient of the Lagrangian with respect to the primal variables cannot be computed, making standard gradient descent-ascent updates inadmissible.

Proxy constraints allow for considering a differentiable surrogate of the constraint when updating the primal variables, while still using the **original non-differentiable constraint for updating the dual variables**. This approach enables the use of gradient-based optimization methods for problems with non-differentiable constraints, **while still ensuring the satisfaction of the original non-differentiable constraints**.

Formally, the optimization problem becomes:

$$
\xstar &\in \argmin{\vx \in \reals^d} \, \, f(\vx) + [\lambdastar]^\top \gtilde(\vx) + [\mustar]^\top \htilde(\vx) \\
\lambdastar, \mustar &\in \argmax{\vlambda \ge \vzero, \vmu} \, \, f(\xstar) + \vlambda^\top \vg(\xstar) + \vmu^\top \vh(\xstar)
$$

where $\vg(\vx) \le \vzero$ and $\vh(\vx) = \vzero$ are the non-differentiable constraints of the problem, and $\gtilde(\vx) \le \vzero$ and $\htilde(\vx) = \vzero$ are differentiable surrogates of $\vg(\vx)$ and $\vh(\vx)$, respectively.

The proxy constraints problem can be solved with the same gradient descent-ascent updates as before, but using the differentiable surrogates $\gtilde(\vx)$ and $\htilde(\vx)$ for the primal updates, and the original non-differentiable constraints $\vg(\vx)$ and $\vh(\vx)$ for the dual updates.

**Cooper** supports proxy constraints when a `strict_violation` is provided in the {py:class}`~cooper.constraints.ConstraintState`. Here, `strict_violation` corresponds to the violation of the original non-differentiable constraint, while `violation` represents the violation of the differentiable surrogate.

While proxy constraints are mainly motivated to solve problems with non-differentiable constraints, **Cooper**'s implementation allows for more general use cases. For example, if computing the exact constraint violation is expensive, you can use a cheaper-to-compute surrogate for the more frequent primal updates, while updating the dual variables with the exact constraint violation albeit less frequently (see TODO tutorial).
