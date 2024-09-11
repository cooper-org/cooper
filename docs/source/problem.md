(cmp)=

# Constrained Minimization Problems

We consider constrained minimization problems (CMPs) expressed as:

$$
\min_{\boldsymbol{x} \in \mathbb{R}^d} & \,\, f(\boldsymbol{x}) \\ \text{s.t. }
& \,\, \boldsymbol{g}(\boldsymbol{x}) \le \boldsymbol{0} \\ & \,\, \boldsymbol{h}(\boldsymbol{x}) = \boldsymbol{0}
$$

Note that $f$ is a scalar-valued function, whereas $\boldsymbol{g}$ and $\boldsymbol{h}$ are vector-valued functions. We group together all the inequality constraints in $\boldsymbol{g}$ and all the equality constraints in $\boldsymbol{h}$.
In other words, a component function $h_i(x)$ corresponds to the scalar constraint
$h_i(\boldsymbol{x}) = 0$.

:::{admonition} Conventions and terminology

- We refer to $f$ as the **loss** or **objective** to be minimized.
- We adopt the convention $g(\boldsymbol{x}) \le 0$ for inequality constraints and $h(\boldsymbol{x}) = 0$ for equality constraints. If your constraints are different, for example $g(\boldsymbol{x}) \ge \epsilon$, you should provide **Cooper** with $\epsilon - g(\boldsymbol{x}) \le 0$.
- We use the term **constraint violation** to refer to $\boldsymbol{g}(\boldsymbol{x})$ and $\boldsymbol{h}(\boldsymbol{x})$. Equality constraints $h(x)$ are satisfied *only* when their defect is zero. On the other hand, a *negative* defect for an inequality constraint $g(x)$ means that the constraint is *strictly* satisfied; while a *positive* defect means that the inequality constraint is being violated.
:::

An approach for solving general non-convex CMPs is to formulate their Lagrangian and find a min-max point:

$$
\boldsymbol{x}^*, \boldsymbol{\lambda}^*, \boldsymbol{\mu}^* = \underset{\boldsymbol{x}}{\text{argmin}} \, \, \underset{\boldsymbol{\lambda} \ge \boldsymbol{0}, \boldsymbol{\mu}}{\text{argmax}} \, \, \mathcal{L}(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\mu})
$$

where $\mathcal{L}(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\mu}) = f(\boldsymbol{x}) + \boldsymbol{\lambda}^T \boldsymbol{g}(\boldsymbol{x}) + \boldsymbol{\mu}^T \boldsymbol{h}(\boldsymbol{x})$, and $\boldsymbol{\lambda}$ and $\boldsymbol{\mu}$ are the Lagrange multipliers associated with the inequality and equality constraints, respectively.
We refer to $\boldsymbol{x}$ as the **primal variables** of the CMP, and $\boldsymbol{\lambda}$ and $\boldsymbol{\mu}$ as the **dual variables**.

An argmin-argmax point of the Lagrangian corresponds to a solution of the original CMP {cite:p}`boyd2004convex`. We refer to finding such a point as the **Lagrangian approach** to solving a constrained minimization problem. **Cooper** is primarily designed to solve CMPs using the Lagrangian approach, but it also implements alternative formulations such as the {py:class}`~cooper.formulation.AugmentedLagrangianFormulation` (see {doc}`formulations`).

An approach for finding min-max points of the Lagrangian is doing gradient descent on the primal variables and gradient ascent on the dual variables. **Cooper** leverages PyTorch's automatic differentiation framework to do this efficiently. **Cooper** also implements variants of gradient descent-ascent such as the {py:class}`~cooper.optim.Extragradient` {cite:p}`korpelevich1976extragradient` (see {doc}`optimizers`).

:::{warning}
**Cooper** is primarily oriented towards **non-convex** CMPs that arise
in many deep learning applications. That is, problems for which one of
the functions $f, \boldsymbol{g}, \boldsymbol{h}$ is non-convex. While the techniques
implemented in **Cooper** are applicable to convex problems as well, we
recommend using specialized solvers for convex optimization problems whenever
possible.
:::

In order to express CMPs, we will define the following objects:
- {py:class}`~cooper.constraints.Constraint`: represents a group of constraints, either equality or inequality.
- {py:class}`~cooper.ConstrainedMinimizationProblem`: represents the constrained minimization problem itself. It must include a method `compute_cmp_state` that computes the loss and constraints at a given point.

Moreover, in order to package the values of the loss and constraints, we will define the following objects:
- {py:class}`~cooper.constraints.ConstraintState`: represents the state of a {py:class}`~cooper.constraints.Constraint` by packaging its violation.
- {py:class}`~cooper.CMPState`: represents the state of a CMP at a given point. It contains the values of the loss and {py:class}`~cooper.constraints.ConstraintState` objects for some or all of its associated constraints.

## Example

The example below illustrates the main steps that need to be carried out to
define a {py:class}`~cooper.ConstrainedMinimizationProblem` class. In this

1. *\[Line 4\]* Define a custom class which inherits from {py:class}`~cooper.ConstrainedMinimizationProblem`.
2. *\[Line 6\]* Define a multiplier object for the constraints.
3. *\[Line 8\]* Define the constraint object.
4. *\[Line 10\]* Implement the `compute_cmp_state` method that computes the loss and constraints.
5. *\[Line 12\]* Return the information about the loss and constraints packaged into a {py:class}`~cooper.CMPState`.
6. *\[Line 18\]* (Optional) Modularize the code to allow for evaluating the constraints **only**. This is useful for optimization algorithms that sometimes need to evaluate the constraints without computing the loss.

```{code-block} python
:emphasize-lines: 4,10,14,18,20
:linenos: true

import torch
import cooper

class MyCMP(cooper.ConstrainedMinimizationProblem):
    def __init__(self):
        super().__init__()
        multiplier = cooper.multipliers.DenseMultiplier(num_constraints=..., device=...)
        # By default, constraints are built using `formulation_type=cooper.LagrangianFormulation`
        self.constraint = cooper.Constraint(
            multiplier=multiplier, constraint_type=cooper.ConstraintType.INEQUALITY
        )

    def compute_cmp_state(self, model, inputs, targets):
        loss = ...
        cmp_state = self.compute_violations(model, inputs, targets)
        cmp_state.loss = loss

        return cmp_state

    def compute_violations(self, model, inputs, targets):
        # This method is optional. It allows for evaluating the constraints without computing the loss.
        violation = ... # ensure that the constraint follows the convention "g <= 0"
        constraint_state = cooper.ConstraintState(violation=...)
        observed_constraints = {self.constraint: constraint_state}

        return cooper.CMPState(loss=None, observed_constraints=observed_constraints)
```


## Constraints

{py:class}`~cooper.constraints.Constraint` objects are used to group similar constraints together. While it is possible to have multiple constraints represented by the same {py:class}`~cooper.constraints.Constraint` object, they must share the same type (i.e., all equality or all inequality constraints) and all must be handled through the same {py:class}`~cooper.formulation.Formulation` (for example, all with a Lagrangian formulation). For combining different types of constraints or formulations, you should use separate {py:class}`~cooper.constraints.Constraint` objects.

```{eval-rst}
.. currentmodule:: cooper.constraints
```


```{eval-rst}
.. autoclass:: Constraint
    :members: as_tuple
```

In their simplest form, {py:class}`~cooper.constraints.ConstraintState` objects simply contain the value of the constraint violation. However, they can be extended to enable extra functionality:
- **Sampled constraints**: if not all violations of a {py:class}`Constraint` are observed at every step, you can still use **Cooper** by providing the observed constraint violations in the {py:class}`~cooper.constraints.ConstraintState`. To do this, provide only the observed violations in `violation`, their corresponding indices in `constraint_features`, and make sure that you are using an {py:class}`~cooper.multipliers.IndexedMultiplier` as the multiplier associated with the constraint. **Cooper** will then know which entries to consider when computing contributions of the constraint to the Lagrangian, and which to ignore.
- **Implicit parameterization of the Lagrange multipliers** {cite:p}`narasimhan2020multiplier`: similar to the sampled constraints case, you can use an implicit parameterization for the Lagrange multipliers (a neural network, for example). In this case, the `constraint_features` must contain the input features to the Lagrange multiplier model associated with the evaluated constraints. Implicit multipliers are discussed in more detail in {doc}`multipliers`.
- **Proxy constraints** {cite:p}`cotter2019proxy`: in some settings, it is desirable to use different constraint violations for updating the primal and dual variables. This can be achieved by a `violation`, which will be used for updating the primal variables, and a `strict_violation`, which will be used for updating the dual variables. When following this approach, ensure that the `violation` is differentiable with respect to the primal variables. Note that proxy constraints can be used in conjunction with sampled constraints and implicit parameterization of the Lagrange multipliers, by providing both `constraint_features` and `strict_constraint_features`.

```{eval-rst}
.. autoclass:: ConstraintState
    :members: as_tuple
```


## CMP objects

```{eval-rst}
.. currentmodule:: cooper
```

{py:class}`ConstrainedMinimizationProblem` objects must be implemented by the user, as exemplified in the [example](#example) above.

```{eval-rst}
.. autoclass:: ConstrainedMinimizationProblem
    :members:
```

## CMPState

We represent computationally the "state" of a CMP using a {py:class}`CMPState`
object. A {py:class}`CMPState` is a dataclass containing the information about the
loss and the equality/inequality violations at a given point $\boldsymbol{x}$. The constraints included in the `CMPState` must be passed as a dictionary, where the keys are the {py:class}`Constraint` objects and the values are the associated {py:class}`ConstraintState` objects.

:::{admonition} Stochastic estimates in `CMPState`
:class: important

In problems for which computing the loss or constraints exactly is prohibitively
expensive, the {py:class}`CMPState` may contain stochastic estimates of the
loss/constraints. For example, this is the case when the loss corresponds to a
sum over a large number of terms, such as training examples. In this case, the
loss and constraints may be estimated using mini-batches of data.

Note that, just as in the unconstrained case, these approximations can
entail a compromise in the stability of the optimization process.
:::

```{eval-rst}
.. autoclass:: CMPState
    :members: as_tuple
```
