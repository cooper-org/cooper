(cmp)=

# Constrained Minimization Problems

We consider constrained minimization problems (CMPs) of the form:

$$
\min_{\vx \in \reals^d} & \,\, f(\vx) \\ \text{s.t. }
& \,\, \vg(\vx) \le \vzero \\ & \,\, \vh(\vx) = \vzero
$$

See {ref}`here<overview>` for a brief introduction to constrained optimization. In this section, we will discuss how to represent CMPs using **Cooper**. To do this, consider the following objects:
- {py:class}`~cooper.constraints.Constraint`: represents a group of either equality or inequality constraints.
- {py:class}`~cooper.ConstrainedMinimizationProblem`: represents the constrained minimization problem itself. It must include a method {py:meth}`~cooper.ConstrainedMinimizationProblem.compute_cmp_state` that computes the loss and constraint violations at a given point.

Moreover, in order to package the values of the loss and constraints, we will define the following objects:
- {py:class}`~cooper.constraints.ConstraintState`: contains the value of the constraint violation at the given iterate.
- {py:class}`~cooper.CMPState`: contains the values of the loss and {py:class}`~cooper.constraints.ConstraintState` objects for (some or all of) its associated constraints.

## Example

The example below illustrates the required steps for defining a {py:class}`~cooper.ConstrainedMinimizationProblem` class for your problem. For simplicity, we illustrate the case of a single (possibly multi-dimensional) inequality constraint.
1. **\[Line 4\]** Define a custom class which inherits from {py:class}`~cooper.ConstrainedMinimizationProblem`.
2. **\[Line 7\]** Instantiate a multiplier object for the constraint.
3. **\[Lines 9-11\]** Define the constraint object, specifying the constraint type and (optionally) the formulation type.
4. **\[Line 13\]** Implement the {py:meth}`~cooper.ConstrainedMinimizationProblem.compute_cmp_state` method that evaluates the loss and constraints.
5. **\[Line 18\]** Return the information about the loss and constraints packaged into a {py:class}`~cooper.CMPState`.
6. **\[Line 20\]** (Optional) Modularize the code to allow for evaluating the constraints **only**. This is useful for optimization algorithms that sometimes need to evaluate the constraints without computing the loss.

```{code-block} python
:emphasize-lines: 4, 7, 9-11, 13, 18, 20
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
        # This method is optional. It allows for evaluating the constraints without
        # computing the loss.
        violation = ... # ensure that the constraint follows the convention "g <= 0"
        constraint_state = cooper.ConstraintState(violation=...)
        observed_constraints = {self.constraint: constraint_state}

        return cooper.CMPState(loss=None, observed_constraints=observed_constraints)
```

## Loss

The {py:class}`CMPState` dataclass includes the loss, which must be a scalar `torch.Tensor` representing $f(\vx)$, and is used to update the primal variables in the optimization. For feasibility problems, where only constraint satisfaction matters, the loss should be set to `None`.

## Constraints

{py:class}`~cooper.constraints.Constraint` objects are used to group similar constraints together. While it is possible to have multiple constraints represented by the same {py:class}`~cooper.constraints.Constraint` object, they must share the same type (i.e., all equality or all inequality constraints) and all must be handled through the same {py:class}`~cooper.formulations.Formulation` (for example, a {py:class}`~cooper.formulations.LagrangianFormulation`). For problems with different types of constraints or formulations, you should instantiate separate {py:class}`~cooper.constraints.Constraint` objects.


To construct the constraint, instantiate a {py:class}`~cooper.multipliers.Multiplier` object. The {py:class}`~cooper.constraints.Constraint` will be associated to this multiplier, and the observed constraint values will be used to update the multiplier. For more details on multiplier objects, see {doc}`multipliers`.


```{eval-rst}
.. currentmodule:: cooper.constraints
```


```{eval-rst}
.. autoclass:: Constraint
```

## ConstraintStates

In their simplest form, {py:class}`~cooper.constraints.ConstraintState` objects simply contain the value of the constraint violation. However, they can be extended to enable extra functionality, such as constraint sampling, the use of implicit parameterizations for the Lagrange multipliers, and proxy constraints.

### Sampled Constraints

When not all violations of a constraint are observed at each step, **Cooper** can still handle these cases by specifying the observed constraint violations in the {py:class}`ConstraintState`. To set this up:

1. Provide only the observed violations in `violation`.
2. Include their corresponding indices in `constraint_features`.
3. Use an {py:class}`~cooper.multipliers.IndexedMultiplier` as the multiplier associated with the constraint.

This setup allows **Cooper** to selectively account for the specified constraint entries when calculating contributions to the Lagrangian, ignoring those that are not observed. More details on indexed multipliers can be found in the section on {doc}`multipliers`.

>
- **Implicit parameterization of the Lagrange multipliers** {cite:p}`narasimhan2020multiplier`: similar to the sampled constraints case, you can use an implicit parameterization for the Lagrange multipliers (a neural network, for example). In this case, the `constraint_features` must contain the input features to the Lagrange multiplier model associated with the evaluated constraints. Implicit multipliers are discussed in more detail in {doc}`multipliers`.

### Implicit Parameterization of Lagrange Multipliers {cite:p}`narasimhan2020multiplier`

**Cooper** lets you parameterize the Lagrange multipliers implicitly, for instance, by using a neural network. To achieve this:

1. Implement a Lagrange multiplier model by inheriting from {py:class}`~cooper.multipliers.ImplicitMultiplier`. The model's input features will identify the constraint being evaluated, with the output corresponding to the associated Lagrange multiplier.
2. Set the `constraint_features` in the {py:class}`ConstraintState` to include the input features for the Lagrange multiplier model, corresponding to each evaluated constraint.

This enables **Cooper** to compute multipliers based on specified input features and to learn a parametric model for the Lagrange multipliers, rather than defining individual multipliers per constraint. For additional details, refer to {doc}`multipliers`.

### Proxy Constraints {cite:p}`cotter2019proxy`

In some cases, it may be useful to employ different constraint violations for updating the primal and dual variables (see {ref}`here<proxy>` for details). To use proxy constraints in **Cooper**:

1. Specify `violation` for updating the **primal variables**. Ensure that this `violation` is differentiable with respect to the primal variables.
2. Provide `strict_violation` for updating the **dual variables**.

Proxy constraints can be combined with sampled constraints and implicit multipliers. To do so, include both `constraint_features` and `strict_constraint_features` as needed.



```{eval-rst}
.. autoclass:: ConstraintState
```


## CMPs

```{eval-rst}
.. currentmodule:: cooper
```

{py:class}`ConstrainedMinimizationProblem` objects must be implemented by the user, as shown in the [example](#example) above.

```{eval-rst}
.. autoclass:: ConstrainedMinimizationProblem
    :members:
```

## CMPStates

We represent computationally the "state" of a CMP using a {py:class}`CMPState` object. A {py:class}`CMPState` is a dataclass containing the information about the loss and the constraint violations measured at a given point. The constraints included in the {py:class}`CMPState` must be passed as a dictionary, where the keys are the {py:class}`Constraint` objects and the values are the associated {py:class}`ConstraintState` objects.

:::{admonition} Stochastic estimates in {py:class}`CMPState`
:class: important

When computing the loss or constraints exactly is prohibitively expensive,
the {py:class}`CMPState` may include **stochastic estimates**. This is often the case
when mini-batches are used to approximate the loss and constraints.

Note that, just as in the unconstrained case, these approximations can entail a compromise in the stability of the optimization process.
:::

```{eval-rst}
.. autoclass:: CMPState
    :members:
```
