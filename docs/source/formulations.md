(formulations)=

```{eval-rst}
.. currentmodule:: cooper.formulations
```

# Formulations

Once a {ref}`constrained minimization problem (CMP)<cmp>` is defined, various algorithmic approaches can be used to find a solution. This process occurs in two stages: the **formulation** of the optimization problem and the selection of the **optimization algorithm** to solve it.

This section focuses on the formulations of the CMP. In {doc}`optim`, we discuss the algorithms for solving the formulated problem (for example, simultaneous gradient descent-ascent).

The formulations supported by **Cooper** are of the form:

$$
\min_{\vx \in \reals^d} \,\, \max_{\vlambda \ge \vzero, \vmu} \,\, f(\vx) + P(\vg(\vx), \vlambda, \vc_{\vg}) + Q(\vh(\vx), \vmu, \vc_{\vh}),
$$

where $P$ and $Q$ are penalty functions aimed at enforcing the satisfaction of the constraints, with multipliers $\vlambda$ and $\vmu$, and hyper-parameters $\vc_{\vg}$ and $\vc_{\vh}$.

::: {note}
**Cooper**'s framework for formulations supports a wide range of approaches for solving constrained optimization problems, including:

- **The Lagrangian formulation**, by letting:

  $$P(\vg(\vx), \vlambda, \vc_{\vg}) = \vlambda^\top \vg(\vx),$$

  $$Q(\vh(\vx), \vmu, \vc_{\vh}) = \vmu^\top \vh(\vx).$$

- **The Augmented Lagrangian formulation**, by letting:

  $$P(\vg(\vx), \vlambda, \vc_{\vg}) = \vlambda^\top \vg(\vx) + \frac{\vc_{\vg}}{2} ||\texttt{relu}(\vg(\vx))||_2^2,$$

  $$Q(\vh(\vx), \vmu, \vc_{\vh}) = \vmu^\top \vh(\vx) + \frac{\vc_{\vh}}{2} ||\vh(\vx)||_2^2.$$

- **Penalty methods** such as the quadratic penalty method; not currently implemented.

- **Interior-point methods** (not currently implemented).
:::


:::{warning}
**Cooper**'s formulation framework is not exhaustive; for instance, it does not support Sequential Quadratic Programming (SQP).
:::

## Example


To specify your formulation of choice, pass the corresponding class to the `formulation_type` argument of the {py:class}`~cooper.constraints.Constraint` class. For example, to use the Lagrangian formulation, you would do:

```python
my_constraint = cooper.Constraint(
    constraint_type=cooper.ConstraintType.INEQUALITY,
    multiplier=multiplier,
    formulation=cooper.formulations.Lagrangian
)
```

:::{note}
**Cooper** provides flexibility by allowing a single CMP to be solved using *different* formulations. The choice of formulation is linked to the **constraints** rather than the CMP itself. Thus, by creating separate {py:class}`~cooper.constraints.Constraint` objects, each with its own formulation, you can apply different formulations to individual constraints.
:::

## Primal and Dual Lagrangians

To support {ref}`proxy`, **Cooper** formulations internally calculate two Lagrangian terms, $\mathcal{L}_{\text{primal}}$ and $\mathcal{L}_{\text{dual}}$. The primal term, $\mathcal{L}_{\text{primal}}$, is used to update the primal variables, while the dual term, $\mathcal{L}_{\text{dual}}$, is used to update the dual variables.

* The primal term, $\mathcal{L}_{\text{primal}}$, is calculated using the objective function $f(\vx)$ and the differentiable constraint violations $\tilde{\vg}(\vx)$ and $\tilde{\vh}(\vx)$, provided in the `violations` entry of a constraint's {py:class}`~cooper.constraints.ConstraintState`.

* The dual term, $\mathcal{L}_{\text{dual}}$, is based on the (potentially non-differentiable) constraint violations $\vg(\vx)$ and $\vh(\vx)$, provided in the `strict_violations` entry of {py:class}`~cooper.constraints.ConstraintState`. For more details, see {ref}`constraint-state`.


## Lagrangian Formulations

The **Lagrangian** formulation of a CMP is:

$$
\min_{\vx \in \reals^d} \,\, \max_{\vlambda \ge \vzero, \vmu} \,\, \mathcal{L}(\vx, \vlambda, \vmu) = f(\vx) + \vlambda^\top \vg(\vx) + \vmu^\top \vh(\vx),
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
\mathcal{L}_{c}(\vx, \vlambda, \vmu) = f(\vx) + \vlambda^\top \vg(\vx) + \vmu^\top \vh(\vx) + \frac{c}{2} ||\texttt{relu}(\vg(\vx))||_2^2 + \frac{c}{2} ||\vh(\vx)||_2^2,
$$

where $c > 0$ is a penalty coefficient.

The main advantage of the Augmented Lagrangian Method (ALM) over the quadratic penalty method (see Theorem 17.5 in {cite:t}`nocedal2006NumericalOptimization`) is that, under certain assumptions, there exists a finite $\bar{c}$ such that for all $c \geq \bar{c}$, the minimizer of $\mathcal{L}_{c}(\vx, \vlambda, \vmu)$ with respect to $\vx$ corresponds to the solution of the original constrained optimization problem. This allows the algorithm to succeed without the need to unboundedly increase the penalty parameter, a requirement often present in the quadratic penalty method.

To use the Augmented Lagrangian formulation in **Cooper**, first define a penalty coefficient (see {ref}`multipliers` for details):

```python
from cooper.multipliers import DensePenaltyCoefficient

penalty_coefficient = DensePenaltyCoefficient(init=torch.tensor(1.0))
```

Then, pass the {py:class}`~AugmentedLagrangianFunction` class and the `penalty_coefficient` to the {py:class}`~cooper.constraints.Constraint` constructor:

```python
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
\vx_{t+1} &\in \argmin{\vx \in \reals^d} \,\, \mathcal{L}_{c_t}(\vx, \vlambda_t, \vmu_t) \\
\vlambda_{t+1} &= \left[\vlambda_t + {\color{red} c_t} \, \vg(\vx_{\color{red} t+1})\right]_+ \\
\vmu_{t+1} &= \vmu_t + {\color{red} c_t} \, \vh(\vx_{\color{red} t+1}) \\
c_{t+1} &\ge c_t
$$

The Augmented Lagrangian method has the following distinguishing features:
1. The minimization with respect to the primal variables $\vx$ is (usually) solved completely or approximately (in contrast to taking one gradient step).
2. It uses alternating updates (the updated primal iterate $\vx_{t+1}$ is used to update the Lagrange multipliers $\vlambda_{t+1}$).
3. The dual learning rate matches the current value of the penalty coefficient $\eta_{\vlambda} = \eta_{\vmu} = c_t$.

If you wish to use the Augmented Lagrangian **Method**, follow these steps:
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

## Base Formulation Class

If you are interested in implementing your own formulation, you can inherit from the {py:class}`~cooper.formulation.Formulation` abstract base class.

```{eval-rst}
.. autoclass:: Formulation
    :members: expects_penalty_coefficient, compute_contribution_to_primal_lagrangian, compute_contribution_to_dual_lagrangian
```

## Utils

TODO: fix docstrings, say something here.

```{eval-rst}
.. automodule:: cooper.formulations.utils
    :members:
```
