(formulations)=

```{eval-rst}
.. currentmodule:: cooper.formulations
```

# Formulations

Once a {ref}`constrained minimization problem (CMP)<cmp>` is defined:

$$
\min_{\vx \in \reals^d} \,\, f(\vx), \,\, \text{s.t. } \,\, \vg(\vx) \le \vzero, \,\, \vh(\vx) = \vzero,
$$

various algorithmic approaches can be used to find a solution. The process involves two stages: first, selecting a **formulation** for the optimization problem, and second, choosing the **optimization algorithm** to solve that formulated problem.

This section focuses on the **formulations** of the CMP. A formulation is a mathematical representation of the constrained optimization problem, involving a *scalarization* of the objective function and constraints (see Sec. 4.7.4 in {cite:t}`boyd2004convex`). Examples of formulations include the {py:class}`Lagrangian` and {py:class}`AugmentedLagrangian` formulations.

## Formulations in **Cooper**

The formulations supported by **Cooper** are of the form:

$$
\min_{\vx \in \reals^d} \,\, \max_{\vlambda \ge \vzero, \vmu} \,\, \Lag(\vx,\vlambda, \vmu) \triangleq f(\vx) + P(\vg(\vx), \vlambda, \vc_{\vg}) + Q(\vh(\vx), \vmu, \vc_{\vh}),
$$

where $P$ and $Q$ are functions aimed at enforcing the satisfaction of the constraints, with decision variables $\vx$, $\vlambda$ and $\vmu$, and hyper-parameters $\vc_{\vg} \geq 0$ and $\vc_{\vh} \geq 0$.

**Cooper**'s framework for formulations supports a wide range of approaches for solving constrained optimization problems, including:

- **The Lagrangian formulation** {cite}`arrow1958studies`, by letting:

  $$P(\vg(\vx), \vlambda, \vc_{\vg}) = \vlambda^\top \vg(\vx),$$

  $$Q(\vh(\vx), \vmu, \vc_{\vh}) = \vmu^\top \vh(\vx).$$

  This results in the usual Lagrangian function:

  $$\Lag(\vx, \vlambda, \vmu) = f(\vx) + \vlambda^\top \vg(\vx) + \vmu^\top \vh(\vx).$$


- **The Quadratic Penalty Formulation** {cite:t}`nocedal2006NumericalOptimization`, by letting:

  $$P(\vg(\vx), \vlambda, \vc_{\vg}) = \frac{1}{2} \vc_{\vg}^\top \, \texttt{relu}(\vg(\vx))^2,$$
  $$Q(\vh(\vx), \vmu, \vc_{\vh}) = \frac{1}{2} \vc_{\vh}^\top \, \vh(\vx)^2.$$

  This results in the Quadratic Penalty function:

  $$\Lag^{\text{QP}}_{\vc_g, \vc_h}(\vx) = f(\vx) + \frac{1}{2} \vc_{\vg}^\top \, \texttt{relu}(\vg(\vx))^2 + \frac{1}{2} \vc_{\vh}^\top \, \vh(\vx)^2.$$


- **The Augmented Lagrangian formulation** {cite}`bertsekas_1975` combines the Lagrangian function with a quadratic penalty term, yielding:

    $$\Lag_{\vc_g, \vc_h}(\vx, \vlambda, \vmu) = \Lag(\vx, \vlambda, \vmu) + \frac{1}{2} \vc_{\vg}^\top \, \texttt{relu}(\vg(\vx))^2 + \frac{1}{2} \vc_{\vh}^\top \, \vh(\vx)^2.$$

For other formulations that could be implemented in future versions of **Cooper**, see {ref}`contributing`.

:::{warning}
Sequential Quadratic Programming (SQP) is not supported under **Cooper**'s formulation framework.
:::

### Multipliers
The decision variables $\vlambda \geq \vzero$ and $\vmu$, refered to as *Lagrange multipliers* or *dual variables*, are used to enforce the constraints. Consider the Lagrangian formulation:

$$
\Lag(\vx, \vlambda, \vmu) = f(\vx) + \vlambda^\top \vg(\vx) + \vmu^\top \vh(\vx).
$$

The maximization of the Lagrangian over $\vlambda$ and $\vmu$ **enforces** the constraints by assigning an objective value of $\infty$ to infeasible points:

$$
\min_{\vx \in \reals^d} \,\, \max_{\vlambda \ge \vzero, \vmu} \,\, \Lag(\vx, \vlambda, \vmu) = \min_{\vx \in \reals^d} \,\, \begin{cases} f(\vx), & \text{if } \vg(\vx) \le \vzero, \vh(\vx) = \vzero, \\ \infty, & \text{otherwise}. \end{cases}
$$

### Penalty Coefficients
The hyper-parameters $\vc_{\vg}$ and $\vc_{\vh}$, refered to as *penalty coefficients*, are used to penalize constraint violations. Consider a quadratic penalty formulation:

$$
    \min_{\vx \in \reals^d} \,\, f(\vx) + \frac{1}{2} \vc_{\vg}^\top \, \texttt{relu}(\vg(\vx))^2 + \frac{1}{2} \vc_{\vh}^\top \, \vh(\vx)^2.
$$

As $\vc_{\vg}, \vc_{\vh} \rightarrow \infty$, a solution to the quadratic penalty problem approaches a solution to the original constrained optimization problem.

For details on how Lagrange multipliers and penalty coefficients are implemented in **Cooper**, see {ref}`multipliers`.


### Solving a Formulation
In {doc}`optim`, we discuss the algorithms used to solve the formulated problem, such as simultaneous gradient descent-ascent ({py:class}`~cooper.optim.SimultaneousOptimizer`).

### Primal and Dual Lagrangians

To support {ref}`proxy`, **Cooper** formulations internally calculate two Lagrangian terms, $\Lag_{\text{primal}}$ and $\Lag_{\text{dual}}$. The *primal* term, $\Lag_{\text{primal}}$, considers differentiable constraint violations $\tilde{\vg}(\vx)$ and $\tilde{\vh}(\vx)$, while the *dual* term, $\Lag_{\text{dual}}$, considers the (potentially non-differentiable) true constraint violations $\vg(\vx)$ and $\vh(\vx)$.
For details on how **Cooper** represents these constraint violations, see {ref}`constraint-state`.


## Example

To specify your formulation of choice, pass the corresponding class to the `formulation_type` argument of the {py:class}`~cooper.constraints.Constraint` class. For example, to use the {py:class}`Lagrangian` formulation, you would write:

```{code-block} python
:emphasize-lines: 4

my_constraint = cooper.Constraint(
    constraint_type=cooper.ConstraintType.INEQUALITY,
    multiplier=multiplier,
    formulation=cooper.formulations.Lagrangian
)
```

:::{note}
**Cooper** provides flexibility by allowing a single CMP to be solved using *different* formulations. The choice of formulation is linked to the **constraints** rather than the CMP itself. Thus, by creating separate {py:class}`~cooper.constraints.Constraint` objects, each with its own formulation, you can apply different formulations to individual constraints.
:::


## Lagrangian Formulations

The **Lagrangian** formulation of a CMP is:

$$
\min_{\vx \in \reals^d} \,\, \max_{\vlambda \ge \vzero, \vmu} \,\, \Lag(\vx, \vlambda, \vmu) = f(\vx) + \vlambda^\top \vg(\vx) + \vmu^\top \vh(\vx),
$$

where $\vlambda$ and $\vmu$ are the Lagrange multipliers or **dual variables** associated with the inequality and equality constraints, respectively.


:::{warning}
There is no guarantee that a general nonconvex constrained optimization problem admits optimal Lagrange multipliers at its solution, $\xstar$. In such cases, attempting to find $\xstar$, $\lambdastar$, and $\mustar$ as an argmin-argmax point of the Lagrangian is ineffective, as $\lambdastar$ and $\mustar$ may not exist.

See {cite:t}`boyd2004convex` for conditions under which Lagrange multipliers are guaranteed to exist.
:::


```{eval-rst}
.. autoclass:: Lagrangian
    :members:
```

(augmented-lagrangian-formulation)=

## Augmented Lagrangian Formulations

The **Augmented Lagrangian** function is a generalization of the Lagrangian function that includes a quadratic penalty term on the constraint violations:

$$
\Lag_{c}(\vx, \vlambda, \vmu) = f(\vx) + \vlambda^\top \vg(\vx) + \vmu^\top \vh(\vx) + \frac{c}{2} ||\texttt{relu}(\vg(\vx))||_2^2 + \frac{c}{2} ||\vh(\vx)||_2^2,
$$

where $c > 0$ is a penalty coefficient.

The main advantage of the Augmented Lagrangian Method (ALM) over the quadratic penalty method (see Theorem 17.5 in {cite:t}`nocedal2006NumericalOptimization`) is that, under certain assumptions, there exists a finite $\bar{c}$ such that for all $c \geq \bar{c}$, the minimizer of $\Lag_{c}(\vx, \vlambda, \vmu)$ with respect to $\vx$ corresponds to the solution of the original constrained optimization problem. This allows the algorithm to succeed without requiring $c \rightarrow \infty$, a requirement often present in the quadratic penalty method.

To use the Augmented Lagrangian formulation in **Cooper**, first define a penalty coefficient (see {ref}`multipliers` for details):

```python
from cooper.multipliers import DensePenaltyCoefficient

penalty_coefficient = DensePenaltyCoefficient(init=torch.tensor(1.0))
```

Then, pass the {py:class}`~AugmentedLagrangianFunction` class and the `penalty_coefficient` to the {py:class}`~cooper.constraints.Constraint` constructor:

```{code-block} python
:emphasize-lines: 4

my_constraint = cooper.Constraint(
    constraint_type=cooper.ConstraintType.INEQUALITY,
    multiplier=multiplier,
    formulation=cooper.formulations.AugmentedLagrangianFunction,
    penalty_coefficient=penalty_coefficient,
)
```

:::{note}
**Cooper** supports vector-valued penalty coefficients that match the size of a constraint. This can be done by passing a tensor of coefficients to the `init` argument of a {py:class}`~cooper.multipliers.PenaltyCoefficient`, where each element corresponds to a penalty coefficient for an individual constraint.
:::

Since it is often desirable to increase the penalty coefficient over the optimization process, **Cooper** provides a scheduler mechanism to do so. For more information, see {ref}`coefficient_updaters`.

:::{warning}
We make a distinction between the Augmented Lagrangian *formulation* ({py:class}`~AugmentedLagrangianFunction`) and the Augmented Lagrangian *method* ({py:class}`~AugmentedLagrangian`, $\S$4.2.1 in {cite:t}`bertsekas1999NonlinearProgramming`). The Augmented Lagrangian method is a specific optimization algorithm over the Augmented Lagrangian function defined above, with the following update rules:

$$
\vx_{t+1} &\in \argmin{\vx \in \reals^d} \,\, \Lag_{c_t}(\vx, \vlambda_t, \vmu_t) \\
\vlambda_{t+1} &= \left[\vlambda_t + {\color{red} c_t} \, \vg(\vx_{\color{red} t+1})\right]_+ \\
\vmu_{t+1} &= \vmu_t + {\color{red} c_t} \, \vh(\vx_{\color{red} t+1}) \\
c_{t+1} &\ge c_t
$$

The Augmented Lagrangian method has the following distinguishing features:
1. The minimization with respect to the primal variables $\vx$ is (usually) solved completely or approximately (in contrast to taking one gradient step).
2. It uses alternating updates (the *updated* primal iterate $\vx_{t+1}$ is used to update the Lagrange multipliers $\vlambda_{t+1}$).
3. The dual learning rates match the current value of the penalty coefficient $\eta_{\vlambda} = \eta_{\vmu} = c_t$.

If you want to use the Augmented Lagrangian *formulation* in **Cooper**, use the {py:class}`~AugmentedLagrangianFunction` formulation. If you intend to use the Augmented Lagrangian **Method**, follow these steps:
1. Use an {py:class}`~AugmentedLagrangian` formulation. This formulation automatically ensures that the dual learning rate is multiplied by the current value of the penalty coefficient.
2. Use {py:class}`torch.optim.SGD` with `lr=1.0` as the optimizer for the dual variables. This ensures that the dual learning rate is simply the penalty coefficient, as opposed `lr * c_t`.
3. Use {py:class}`~cooper.optim.PrimalDualOptimizer` as the constrained optimizer to obtain alternating updates which first update the primal variables and then the dual variables.
4. [Optional] Instead of carrying out a single step of primal optimization for every step of dual optimization, you can carry out multiple primal steps for every dual steps, which more closely approximates the approximate minimization of the Augmented Lagrangian function. For details on how to do this, see {doc}`optim`.

:::


```{eval-rst}
.. autoclass:: AugmentedLagrangianFunction
    :members:
    :exclude-members: compute_contribution_to_primal_lagrangian, compute_contribution_to_dual_lagrangian
```

```{eval-rst}
.. autoclass:: AugmentedLagrangian
    :members:
    :exclude-members: compute_contribution_to_primal_lagrangian, compute_contribution_to_dual_lagrangian
```

## Implementing a Custom Formulation

If you are interested in implementing your own formulation, you can inherit from the {py:class}`~cooper.formulation.Formulation` abstract base class.

### Base Formulation Class


```{eval-rst}
.. autoclass:: Formulation
    :members: expects_penalty_coefficient, compute_contribution_to_primal_lagrangian, compute_contribution_to_dual_lagrangian
```

### Utils

**Cooper** provides utility functions to simplify the core computations of the primal and dual Lagrangians. These functions are used internally by the formulations and can be used to implement custom formulations.

```{eval-rst}
.. automodule:: cooper.formulations.utils
    :members:
```
