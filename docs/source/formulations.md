(formulations)=

```{eval-rst}
.. currentmodule:: cooper.formulations
```

# TODO: Lagrangian store

# Formulations

Once equipped with a {ref}`constrained minimization problem (CMP)<cmp>`, several algorithmic approaches can be adopted to find a solution. These occur in two stages: the **formulation** of the optimization problem and the choice of the **optimization algorithm** to solve it.

This section focuses on formulations of the CMP. {ref}`Here<optim>` we discuss the algorithms for solving the formulated problem (e.g., simultaneous gradient descent-ascent).

The formulations supported by **Cooper** are of the form:

$$
\min_{\vx \in \reals^d} \,\, \max_{\vlambda \ge \vzero, \vmu} \,\, f(\vx) + P(\vg(\vx), \vlambda, \vc_{\vg}) + Q(\vh(\vx), \vmu, \vc_{\vh}),
$$

where $P$ and $Q$ are penalty functions aimed at enforcing the satisfaction of the constraints, with parameters $\vlambda$ and $\vmu$, and hyper-parameters $\vc_{\vg}$ and $\vc_{\vh}$.


:::{warning}
**Cooper**'s framework for formulations supports a wide range of approaches for solving constrained optimization problems, including:
- The Lagrangian (with $\vlambda$ and $\vmu$ as Lagrange multipliers)
- The Augmented Lagrangian (with $\vc_{\vg}$ and $\vc_{\vh}$ as penalty coefficients)
- Penalty methods (e.g., the quadratic penalty method; not currently implemented)
- Interior-point methods (not currently implemented)

However, this framework is not exhaustive and formulations such as Sequential Quadratic Programming (SQPs) are not supported in **Cooper**.
:::

To specify your formulation of choice, pass the corresponding class to the `formulation_type` argument of the {py:class}`~cooper.constraints.Constraint` class:

```python
my_constraint = cooper.Constraint(
    constraint_type=cooper.ConstraintType.INEQUALITY,
    multiplier=multiplier,
    formulation=cooper.LagrangianFormulation
)
```

:::{note}
**Cooper** offers flexibility in that a single CMP can be solved using *different* formulations. Crucially, the choice of formulation is tied to the **constraints**, rather than the CMP itself. By specifying different {py:class}`~cooper.constraints.Constraint`s within a CMP, you can apply different formulations to each.
:::

## Lagrangian Formulations

The **Lagrangian** formulation of a CMP is:

$$
\min_{\vx \in \reals^d} \,\, \max_{\vlambda \ge \vzero, \vmu} \,\, \mathcal{L}(\vx, \vlambda, \vmu) = f(\vx) + \vlambda^\top \vg(\vx) + \vmu^\top \vh(\vx),
$$

where $\vlambda \geq \vzero$ and $\vmu$ are the Lagrange multipliers or **dual variables** associated with the inequality and equality constraints, respectively.


:::{warning}
There is no guarantee that a general nonconvex constrained optimization problem admits optimal Lagrange multipliers at its solution $\xstar$. In such cases, finding $\xstar, \lambdastar, \mustar$ as an argmin-argmax point of the Lagrangian is a futile approach to solve the problem since $\lambdastar$ and $\mustar$ do not exist.

See {cite:t}`boyd2004convex` for conditions under which Lagrange multipliers are guaranteed to exist.
:::


```{eval-rst}
.. autoclass:: LagrangianFormulation
    :members:
```

(augmented-lagrangian-formulation)=

## Augmented Lagrangian Formulation

The Augmented Lagrangian function is a modification of the Lagrangian function that includes a quadratic penalty term on the constraints:

$$
\mathcal{L}_{c}(\vx, \vlambda, \vmu) = f(\vx) + \vlambda^\top \vg(\vx) + \vmu^\top \vh(\vx) + \frac{c}{2} ||\text{max}\{\vzero, \vg(\vx)\}||^2 + \frac{c}{2} ||\vh(\vx)||^2
$$

where $c > 0$ is a penalty coefficient.

TODO: predoc 3

The main advantage of the ALM compared to the quadratic penalty method
(see $\S$ 4.2.1 in {cite:p}`bertsekas1999NonlinearProgramming`) is that
(under some reasonable assumptions), the algorithm can be successful without
requiring the unbounded increase of the penalty parameter sequence $c^t$.
The use of explicit estimates for the Lagrange multipliers contribute to
avoiding the ill-conditioning that is inherent in the quadratic penalty method.

See $\S$ 4.2.1 in {cite:p}`bertsekas1999NonlinearProgramming` and
$\S$ 17 in {cite:p}`nocedal2006NumericalOptimization` for a comprehensive
treatment of the Augmented Lagrangian method.


To use the Augmented Lagrangian formulation in **Cooper**, first define a penalty coefficient (see {ref}`multipliers` for details):

```python
from cooper.multipliers import DensePenaltyCoefficient

penalty_coefficient = DensePenaltyCoefficient(init=1.0)
```

Then, pass the {py:class}`~cooper.formulation.lagrangian.AugmentedLagrangianFormulation` class and the `penalty_coefficient` to the {py:class}`~cooper.constraints.Constraint` constructor:

```python
my_constraint = cooper.Constraint(
    constraint_type=cooper.ConstraintType.INEQUALITY,
    multiplier=multiplier,
    formulation=cooper.AugmentedLagrangianFormulation,
    penalty_coefficient=penalty_coefficient,
)
```

:::{note}
**Cooper** also allows for having different penalty coefficients for different constraints. This can be achieved by passing a tensor of coefficients to the `init` argument of a {py:class}`~cooper.multipliers.PenaltyCoefficient`.
:::

**Cooper** also supports the use of a scheduler for the penalty coefficient. For more information, see {ref}`coefficient_updaters`.

:::{warning}
We make a distinction between the Augmented Lagrangian function/formulation and the Augmented Lagrangian *method* $\S$4.2.1 in {cite:p}`bertsekas1999NonlinearProgramming`. The Augmented Lagrangian method is an optimization algorithm over the Augmented Lagrangian function above:

$$
\vx_{t+1} &\in \argmin{\vx \in \reals^d} \,\, \mathcal{L}_{c_t}(\vx, \vlambda_t, \vmu_t) \\
\vlambda_{t+1} &= \left[\vlambda_t + {\color{red} c_t} \, \vg(\vx_{\color{red} t+1})\right]_+ \\
\vmu_{t+1} &= \vmu_t + {\color{red} c_t} \, \vh(\vx_{\color{red} t+1}) \\
c_{t+1} &\ge c_t
$$

The Augmented Lagrangian method has the following distinguishing features:
- The minimization with respect to the primal variables $\vx$ is (usually) solved completely or approximately (in contrast to taking one gradient step).
- It uses alternating updates (the updated primal iterate $\vx_{t+1}$ is used to update the Lagrange multipliers $\vlambda_{t+1}$).
- The dual learning rate matches the current value of the penalty coefficient $\eta_{\vlambda} = \eta_{\vmu} = c_t$.

If you are interested in using the Augmented Lagrangian method in **Cooper**, use the {py:class}`~cooper.optim.PrimalDualOptimizer` constrained optimizer and ensure that the learning rate of the dual variables is linked to the penalty coefficient by doing:

```python
TODO
```
:::


```{eval-rst}
.. autoclass:: AugmentedLagrangianFormulation
    :members:
```

## Base Formulation Class

If you are interested in implementing your own formulation, you can inherit from the {py:class}`~cooper.formulation.Formulation` abstract base class.

```{eval-rst}
.. autoclass:: Formulation
    :members: compute_contribution_to_primal_lagrangian, compute_contribution_to_dual_lagrangian
```

## Utils

```{eval-rst}
.. automodule:: cooper.formulations.utils
    :members:
```
