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
- We adopt the less-than-or-equal convention $\vg(\vx) \le \vzero$ for **inequality constraints** and $\vh(\vx) = \vzero$ for **equality constraints**. If your constraints are of a different form, for example $\vc(\vx) \ge \epsilon$, you should provide **Cooper** with $\vg(\vx) = \epsilon - \vc(\vx) \le \vzero$.
:::

:::{warning}
We use the term constraint violation to refer to both $\vg(\vx)$ and $\vh(\vx)$. For equality constraints, $\vh(\vx)$ is satisfied only when its violation is zero, i.e., $\vh(\vx) = \vzero$. For inequality constraints, a negative violation of $\vg(\vx)$ indicates the constraint is strictly satisfied (i.e., $\vg(\vx) < \vzero$), whereas a positive violation indicates the constraint is violated (i.e., $\vg(\vx) > \vzero$).

Note that we still refer to $\vg(\vx)$ and $\vh(\vx)$ as "violations" even when the constraint are satisfied. This differs from the convention in some of the optimization literature, which uses the term "violation" to refer to the amount by which a constraint is violated (i.e., $\max\{\vzero, \vg(\vx)\}$ for inequality constraints and $|\vh(\vx)|$ for equality constraints).
:::


We group together all the inequality constraints in $\vg$, and all the equality constraints in $\vh$.
In other words, $f$ is a scalar-valued function, whereas $\vg$ and $\vh$ are vector-valued functions with argument $\vx$.
A component function $h_i(\vx)$ corresponds to the scalar constraint $h_i(\vx) = 0$.


:::{note}
:class: note

In constrained optimization problems, the goal is to find the optimal solution among those that satisfy the constraints. In particular, **constraints are meant to be satisfied, not optimized**.

Therefore, two feasible solutions with the same objective value are not inherently preferable to one another, even if one has smaller constraint violations.

:::


## The Lagrangian Approach

An approach for solving general nonconvex constrained optimization problems is to formulate their Lagrangian and find a min-max point:

$$
\xstar, \lambdastar, \mustar = \argmin{\vx \in \reals^d} \, \, \argmax{\vlambda \ge \vzero, \vmu} \, \, \Lag(\vx, \vlambda, \vmu)
$$

where $\Lag(\vx, \vlambda, \vmu) = f(\vx) + \vlambda^\top \vg(\vx) + \vmu^\top \vh(\vx)$ is the **Lagrangian function** associated with the constrained minimization problem. $\vlambda \geq \vzero$ and $\vmu$ are the **Lagrange multipliers** associated with the inequality and equality constraints, respectively.
We refer to $\vx$ as the **primal variables** of the CMP, and $\vlambda$ and $\vmu$ as the **dual variables**.

:::{note}
$\Lag(\vx,\vlambda, \vmu)$ is concave in $\vlambda$ and $\vmu$ regardless of the convexity properties of $f$, $\vg$, and $\vh$.
:::

An argmin-argmax point of the Lagrangian corresponds to a solution of the original CMP {cite:p}`boyd2004convex`. We refer to finding such a point as the **Lagrangian approach** to solving a constrained minimization problem.

**Cooper** is primarily designed to solve constrained optimization problems using the Lagrangian approach, and it also implements alternative formulations such as the {py:class}`~cooper.formulation.QuadraticPenalty` and {py:class}`~cooper.formulation.AugmentedLagrangian` formulations (see {doc}`formulations`).

:::{admonition} Why does **Cooper** use the Lagrangian approach?
**Cooper** is designed for solving constrained optimization problems that arise in deep learning applications. These problems are often **nonconvex** and **high-dimensional**, and may require **estimating constraints stochastically** from mini-batches of data. The Lagrangian approach is well-suited to these problems for several reasons:
- **Nonconvexity**. The Lagrangian approach does not require the loss or constraints to be convex or follow a specific functional structure, making it applicable to general nonconvex problems. Note that for non-convex problems, min-max points of the Lagrangian may not exist.
>
- **Scalability**. First-order optimization methods, such as gradient descent-ascent, can be used to find min-max points of the Lagrangian. These methods are well-supported by automatic differentiation frameworks such as PyTorch and scale to high-dimensional problems.
\
Moreover, the overhead (relative to unconstrained minimization) of storing and updating the Lagrange multipliers is generally negligible in deep learning problems, where the computational cost of calculating the loss, constraints, and their gradients represents the main bottleneck.
>
- **Stochastic estimates of the constraints**. Gradient-based methods can rely on stochastic estimates of the loss and constraints, making them applicatble to problems where computing the exact loss and constraints is prohibitively expensive.
:::

:::{warning}
**Cooper** is primarily designed for **nonconvex** constrained optimization problems that arise in many deep learning applications. While the techniques implemented in **Cooper** are applicable to convex problems as well, we recommend using specialized solvers for convex optimization problems whenever possible.
:::


## Optimality Conditions

We seek a min-max point of the Lagrangian, $(\xstar, \lambdastar, \mustar)$, which corresponds to a solution of the original constrained minimization problem. A min-max point must satisfy the following conditions:

1. **Minimization with respect to** $\vx$:
   $\xstar$ minimizes $\Lag(\vx, \lambdastar, \mustar)$ over $\vx$. A necessary condition for this is **stationarity**:

   $$
   \nabla_{\vx} \Lag(\xstar, \lambdastar, \mustar) = \vzero.
   $$

2. **s** $\vlambda$:
   $\lambdastar$ maximizes $\Lag(\xstar, \vlambda, \mustar)$ over $\vlambda$ for $\vlambda \geq \vzero$. This implies that for each $i$, either:
   - $\lambda^*_i = 0$, or
   - $\lambda^*_i > 0$ and $\nabla_{\lambda_i} \Lag(\xstar, \vlambda, \mustar) = g_i(\xstar) \leq 0$.

    It can be shown that in both cases, $g_i(\xstar) \leq 0$, ensuring **feasibility**. These conditions can be rewritten as:

    - **Complementary slackness**:

    $$ \lambdastar \circ \vg(\xstar) = \vzero. $$
    - **Feasibility**:

    $$ \vg(\xstar) \leq \vzero. $$


3. **Maximization with respect to** $\vmu$:
   $\mustar$ maximizes $\Lag(\xstar, \lambdastar, \vmu)$ over $\vmu$. A necessary condition for this is:

   $$
   \nabla_{\vmu} \Lag(\xstar, \lambdastar, \mustar) = \vh(\xstar) = \vzero.
   $$

These conditions are equivalent to the **Karush-Kuhn-Tucker (KKT) conditions** for the constrained minimization problem {cite:p}`boyd2004convex`.

:::{note}
:class: note

When solving problems numerically, a common definition of approximate stationarity requires $\xstar$ to be:
(i) feasible, and
(ii) have the norm of the Lagrangian gradient bounded by some $\epsilon > 0$:

$$
\| \nabla_{\vx} \Lag(\xstar, \lambdastar, \mustar) \| \leq \epsilon.
$$


:::

### **Sufficient Conditions for Optimality**
A sufficient condition for optimality is:

$$
\nabla_{\vx,\vx}^2 \Lag(\xstar, \lambdastar, \mustar) \succeq \vzero,
$$

where $\nabla_{\vx,\vx}^2 \Lag(\xstar, \lambdastar, \mustar)$ is the Hessian of the Lagrangian with respect to $\vx$. This ensures that the recovered stationary point is a local minimum of the Lagrangian rather than a saddle point or local maximum.


### **Assessing Convergence in Practice**
In practice, convergence is characterized by:

1. **Feasibility**
2. A **norm of the Lagrangian gradient approaching zero**.
3. **Complementary slackness**, where inequality constraints are either:
   - Strictly satisfied ($g_i(\xstar) < 0$) with a zero Lagrange multiplier ($\lambda^*_i = 0$), or
   - Active-satisfied ($g_i(\xstar) = 0$) with a strictly positive Lagrange multiplier ($\lambda^*_i > 0$)

## Solving Min-max Optimization Problems

A simple approach for finding min-max points of the Lagrangian is doing gradient _descent_ on the primal variables and gradient _ascent_ on the dual variables. Simultaneous **gradient descent-ascent** involves the following updates:

$$
\vx_{t+1} &= \vx_t - \eta_{\vx} \nabla_{\vx} \Lag(\vx_t, \vlambda_t, \vmu_t) \\
\vlambda_{t+1} &= \left [ \vlambda_t + \eta_{\vlambda} \nabla_{\vlambda} \Lag(\vx_t, \vlambda_t, \vmu_t) \right ]_+ \\
\vmu_{t+1} &= \vmu_t + \eta_{\vmu} \nabla_{\vmu} \Lag(\vx_t, \vlambda_t, \vmu_t)
$$

where $\eta_{\vx}, \eta_{\vlambda}, \eta_{\vmu}$ are the step-sizes for the primal and dual variables. The projection operator $[\cdot]_+$ ensures that the dual variables associated with the inequality constraints remain non-negative.

Plugging in the gradients of the Lagrangian, we get the following updates:

$$
\vx_{t+1} &= \vx_t - \eta_{\vx} \left [ \nabla_{\vx} f(\vx_t) + \vlambda_t^\top \nabla_{\vx} \vg(\vx_t) + \vmu_t^\top \nabla_{\vx} \vh(\vx_t) \right ] \\
\vlambda_{t+1} &= \left [ \vlambda_t + \eta_{\vlambda} \vg(\vx_t) \right ]_+ \\
\vmu_{t+1} &= \vmu_t + \eta_{\vmu} \vh(\vx_t)
$$

The primal updates follow a linear combination of the gradients of the loss and constraints, with the coefficients corresponding to the Lagrange multipliers. Larger values of a Lagrange multiplier result in a stronger influence of the corresponding constraint on the primal updates, promoting feasibility. Conversely, smaller values (or zero) reduce the influence of the constraint, prioritizing loss reduction.

:::{note}
:class: tip

Note that, unlike in standard minimization, the objective function $f$ may not decrease monotonically during the optimization process, as the emphasis on satisfying constraints may sometimes conflict with the goal of reducing the loss.

For further practical intuitions, see the **Cooper** [FAQ](#faq).

:::

On the other hand, the dual updates accumulate the constraint violations. Together with the primal updates, these encourage the constraints to be satisfied:
- **Inequality constraints**: When a constraint is violated ($\vg(\vx) > \vzero$), the corresponding Lagrange multiplier increases to penalize the violation. If the constraint is strictly satisfied ($\vg(\vx) < \vzero$), the multiplier decreases, allowing the focus to shift toward loss reduction.
- **Equality constraints**: For a positive (resp. negative) violation, the Lagrange multiplier increases (resp. decreases), encouraging a decrease (resp. increase) in the violation. The multiplier stabilizes when the constraint is satisfied ($\vh(\vx) = \vzero$).

**Cooper** leverages PyTorch's automatic differentiation framework to efficiently perform gradient-based optimization of the Lagrangian.
**Cooper** supports simultaneous gradient descent-ascent, as well as other variants like alternating gradient descent-ascent and the {py:class}`~cooper.optim.Extragradient` method {cite:p}`korpelevich1976extragradient`.
For more details on the available methods, see the {doc}`optim` module.

With **Cooper**, you can specify custom {py:class}`~torch.optim.Optimizer` objects for the primal and dual updates, allowing you to apply optimization techniques such as Adam, popular when training deep neural networks.



(proxy)=
## Non-differentiable Constraints

{cite:t}`cotter2019proxy` introduced the concept of **proxy constraints** to address problems with non-differentiable constraints. In these cases, the gradient of the Lagrangian with respect to the primal variables is not defined, making standard gradient descent-ascent inadmissible.

Proxy constraints consider a differentiable _surrogate_ of the constraint when updating the primal variables, while still using the **original non-differentiable constraint for updating the dual variables**. This approach enables the use of gradient-based optimization methods for problems with non-differentiable constraints, **while still ensuring the satisfaction of the original non-differentiable constraints**.

Formally, the optimization problem becomes:

$$
\xstar &\in \argmin{\vx \in \reals^d} \, \, f(\vx) + [\lambdastar]^\top \gtilde(\vx) + [\mustar]^\top \htilde(\vx) \\
\lambdastar, \mustar &\in \argmax{\vlambda \ge \vzero, \vmu} \, \, f(\xstar) + \vlambda^\top \vg(\xstar) + \vmu^\top \vh(\xstar)
$$

where $\vg(\vx) \le \vzero$ and $\vh(\vx) = \vzero$ are the non-differentiable constraints of the problem, and $\gtilde(\vx) \le \vzero$ and $\htilde(\vx) = \vzero$ are differentiable surrogates of $\vg(\vx)$ and $\vh(\vx)$, respectively.

The proxy constraints problem can be solved with the same gradient descent-ascent updates as before, but using the (gradients of the) differentiable surrogates $\gtilde(\vx)$ and $\htilde(\vx)$ for the primal updates, and the original non-differentiable constraints $\vg(\vx)$ and $\vh(\vx)$ for the dual updates.

**Cooper** supports proxy constraints through the `strict_violation` argument of a {py:class}`~cooper.constraints.ConstraintState`. When `strict_violation` is provided, it is interpreted as the violation of the original non-differentiable constraint, while `violation` is interpreted as the violation of the differentiable surrogate.

While proxy constraints are mainly motivated to solve problems with non-differentiable constraints, **Cooper**'s implementation allows for more general use cases. For example, if computing the exact constraint violation is expensive, you can use a cheaper-to-compute surrogate for the more frequent primal updates, while updating the dual variables with the exact constraint violation albeit less frequently. This behavior can be accomplished via the  `contributes_to_primal_update` and `contributes_to_dual_update` flags of the {py:class}`~cooper.constraints.ConstraintState` class. An example is provided in this [tutorial](notebooks/plot_infrequent_true_constraint).
