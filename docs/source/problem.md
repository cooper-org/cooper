(cmp)=

# Constrained Minimization Problems

We consider constrained minimization problems (CMPs) of the form:

$$
\min_{\vx \in \reals^d} & \,\, f(\vx) \\ \text{s.t. }
& \,\, \vg(\vx) \le \vzero \\ & \,\, \vh(\vx) = \vzero
$$

See {ref}`here<overview>` for a brief introduction to constrained optimization. In this section, we will discuss how to represent CMPs using **Cooper**. To do this, consider the following objects:
- {py:class}`~cooper.constraints.Constraint`: represents a group of either equality or inequality constraints.
- {py:class}`~cooper.cmp.ConstrainedMinimizationProblem` (hereafter {py:class}`CMP<cooper.cmp.ConstrainedMinimizationProblem>` in short): represents the constrained minimization problem itself. It must include a method {py:meth}`CMP.compute_cmp_state<cooper.ConstrainedMinimizationProblem.compute_cmp_state>` that computes the loss and constraint violations at a given point.

Moreover, in order to package the values of the loss and constraints, we will define the following objects:
- {py:class}`~cooper.constraints.ConstraintState`: contains the value of the constraint violation at the given iterate.
- {py:class}`~cooper.CMPState`: contains the values of the loss and {py:class}`~cooper.constraints.ConstraintState` objects for (some or all of) its associated constraints.

## Example

The example below illustrates the required steps for defining a {py:class}`CMP<cooper.cmp.ConstrainedMinimizationProblem>` class for your problem. For simplicity, we illustrate the case of a single (possibly multi-dimensional) inequality constraint.
1. **\[Line 4\]** Define a custom class which inherits from {py:class}`CMP<cooper.cmp.ConstrainedMinimizationProblem>`.
2. **\[Line 7\]** Instantiate a multiplier object for the constraint.
3. **\[Lines 9-11\]** Define the constraint object, specifying the constraint type and (optionally) the formulation type. Note that {py:class}`~cooper.constraints.Constraint`s must be "registered" as attributes of the {py:class}`CMP<cooper.cmp.ConstrainedMinimizationProblem>`.
4. **\[Line 13\]** Implement the {py:meth}`CMP.compute_cmp_state<cooper.ConstrainedMinimizationProblem.compute_cmp_state>` method that evaluates the loss and constraints.
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
        # By default, constraints are built using `formulation_type=cooper.formulations.Lagrangian`
        self.my_constraint = cooper.Constraint(
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
        observed_constraints = {self.my_constraint: constraint_state}

        return cooper.CMPState(loss=None, observed_constraints=observed_constraints)
```

## Loss

The {py:class}`~cooper.CMPState` dataclass includes the loss, which must be a scalar {py:class}`torch.Tensor` representing $f(\vx)$, and is used to update the primal variables in the optimization. For feasibility problems, where only constraint satisfaction matters, the loss should be set to `None`.

## Constraints

{py:class}`~cooper.constraints.Constraint` objects allows grouping similar constraints together. Constraints can be classified as either equality or inequality.

It is possible to have multiple constraints represented by the same {py:class}`~cooper.constraints.Constraint` object. However, all constraints under a {py:class}`~cooper.constraints.Constraint` must share the same `constraint_type` (all equality or all inequality constraints) and must be handled using the same `formulation_type` (a subclass of a {py:class}`~cooper.formulations.Formulation`). For problems with different types of constraints or formulations, you should instantiate separate {py:class}`~cooper.constraints.Constraint` objects.

```{eval-rst}
.. autoenum:: cooper.ConstraintType
    :members:
```

:::{admonition} Grouping constraints
:class: note

**Cooper** allows arbitrary groupings of constraints into {py:class}`~cooper.constraints.Constraint` objects.
However, for computational or logging purposes it is sometimes desirable to group constraints according to problem-dependent structures.

For example, consider a problem with $m + n$ constraints.
One could create a _single_ {py:class}`~cooper.constraints.Constraint` object along with a _single_ {py:class}`~cooper.multipliers.Multiplier`.
Alternatively, one could create two ({py:class}`~cooper.constraints.Constraint`, {py:class}`~cooper.multipliers.Multiplier`) pairs for handling the groups of $m$ and $n$ constraints separately.
:::

% Duplicating "Linking constraints and multipliers" on multipliers.md

:::{admonition} Linking constraints and multipliers
:class: hint

{py:class}`~cooper.constraints.Constraint` objects must have an associated {py:class}`~cooper.multipliers.Multiplier` if the problem formulation requires it. see the {py:attr}`~cooper.formulations.Formulation.expects_multiplier` attribute of a {py:class}`~cooper.formulations.Formulation`. To achieve this, a {py:class}`~cooper.multipliers.Multiplier` object should be provided to the {py:class}`~cooper.constraints.Constraint` constructor.

```python
constraint = cooper.Constraint(
    multiplier=multiplier,
    constraint_type=cooper.ConstraintType.INEQUALITY,
    formulation_type=cooper.formulations.Lagrangian,
)
```
:::

(registering-constraints)=
:::{admonition} Registering constraints in a {py:class}`CMP<cooper.cmp.ConstrainedMinimizationProblem>`
:class: warning

When initializing the {py:class}`CMP<cooper.cmp.ConstrainedMinimizationProblem>`, {py:class}`~cooper.constraints.Constraint`s should be defined as attributes.
We refer to this process as "registering" the constraints with the {py:class}`CMP<cooper.cmp.ConstrainedMinimizationProblem>`. Utilities such as {py:meth}`CMP.constraints<cooper.ConstrainedMinimizationProblem.constraints>` and {py:meth}`CMP.named_constraints<cooper.ConstrainedMinimizationProblem.named_constraints>` enable iteration over the registered constraints.

Moreover, the utility method {py:meth}`CMP.dual_parameters<cooper.ConstrainedMinimizationProblem.dual_parameters>` returns the {py:class}`torch.nn.parameter.Parameter`s of the multipliers for the constraints registered with the {py:class}`CMP<cooper.cmp.ConstrainedMinimizationProblem>`. This facilitates the instantiation of the `dual_optimizer` required by the {py:class}`~cooper.optim.ConstrainedOptimizer`. See the [Optim](optim.md#example) module for further details.

```python
class MyCMP(cooper.ConstrainedMinimizationProblem):
    def __init__(self):
        super().__init__()
        self.constraint_1 = cooper.Constraint(...)
        self.constraint_2 = cooper.Constraint(...)

cmp = MyCMP()

# Constraints can also be registered "on the fly"
cmp.constraint_3 = cooper.Constraint(...)

# Inspect constraints known to the CMP
print([constraint_name for constraint_name, constraint in cmp.named_constraints()])

# Output:
# ["constraint_1", "constraint_2", "constraint_3"]
```
:::

```{eval-rst}
.. currentmodule:: cooper.constraints
```

```{eval-rst}
.. autoclass:: Constraint
```

(constraint-state)=

## ConstraintStates

In their simplest form, {py:class}`~cooper.constraints.ConstraintState` objects contain the value of the constraint violation. However, they can be extended to enable extra functionality, such as constraint sampling, the use of [implicit parameterizations](multipliers.md#implicit-parametric-multipliers) for the Lagrange multipliers {cite:p}`narasimhan2020multiplier`, and [proxy constraints](constrained_optimization.md#non-differentiable-constraints) {cite:p}`cotter2019proxy`.

:::
### Constraint Sampling
:::

**Cooper** can handle cases where only a subset of {py:class}`Constraint` violations are observed at each step.
For instance, in problems with a large number of constraints, where assessing feasibility is costly (e.g., requiring evaluation across an entire dataset), it can be more efficient to evaluate only a subset of constraints per iteration.
To do this, specify the subset of observed constraint violations and ensure that an {py:class}`~cooper.multipliers.IndexedMultiplier` is used as the multiplier associated with the constraint.

Consider the case of an $n$-dimensional constraint, where only $m \leq n$ entries are observed. **Cooper** expects the following arguments when instantiating a {py:class}`ConstraintState`:

1. `violation` should be an $m$-dimensional tensor containing only the *observed* violations.
2. `constraint_features` should contain the indices corresponding to the observed entries in `violation`. Note that permutations are allowed, as long as the ordering of `violation` and `constraint_features` is consistent.
3. If provided, `strict_violation` and `strict_constraint_features` should follow the same pattern.

This setup allows **Cooper** to selectively account for the specified constraint entries when calculating contributions to the Lagrangian, ignoring those that are not observed. More details on indexed multipliers can be found in the section on {doc}`multipliers`.

:::
### Implicit Parameterization of Lagrange Multipliers {cite:p}`narasimhan2020multiplier`
:::

**Cooper** enables implicit parameterization of the Lagrange multipliers, for example, by using a neural network.
This allows **Cooper** to compute multipliers based on specified input features and to learn a parametric model for the multipliers, rather than defining them individually for each constraint.
This approach helps scale the Lagrangian approach to problems with numerous (or even infinite) constraints.

To achieve this, follow these steps:

1. Implement a Lagrange multiplier model by inheriting from {py:class}`~cooper.multipliers.ImplicitMultiplier`.
Given input features for a constraint, the model will map the input features to an estimate of the Lagrange multiplier for that constraint.
2. Pass these input features as the `constraint_features` in the {py:class}`ConstraintState` object.

For additional details, refer to {doc}`multipliers`.

:::
### Proxy Constraints {cite:p}`cotter2019proxy`
:::

It is often the case that constrained optimization problems involve non-differentiable constraints. This non-differentiability precludes the use of gradient-based primal-dual optimization methods.
The **proxy constraints** technique introduced by {cite:t}`cotter2019proxy` leverages a differentiable surrogate of the constraint when performing the primal updates, while preserving the original (non-differentiable) constraint for updating the dual variables (see {ref}`here<proxy>` for details).

To use proxy constraints in **Cooper**:

1. Specify `violation` for updating the **primal variables**. Ensure that this `violation` is differentiable with respect to the primal variables.
2. Provide (the potentially non-differentiable) `strict_violation` for updating the **dual variables**.

Proxy constraints can be combined with sampled constraints and implicit multipliers. To do so, include both `constraint_features` and `strict_constraint_features` as needed.

```{eval-rst}
.. autoclass:: ConstraintState
```


## Constrained Minimization Problems (CMPs)

```{eval-rst}
.. currentmodule:: cooper
```

{py:class}`ConstrainedMinimizationProblem` sub-classes must be implemented and instantiated by the user, as shown in the [example](#example) above.

{py:class}`CMP<cooper.cmp.ConstrainedMinimizationProblem>`s represent constrained optimization problems and provide methods to compute the problem’s state, {py:class}`CMPState`, which includes the loss and constraints at a given point. The methods to be implemented are {py:meth}`CMP.compute_cmp_state<ConstrainedMinimizationProblem.compute_cmp_state>` and, optionally,
{py:meth}`CMP.compute_violations<ConstrainedMinimizationProblem.compute_violations>`.

Additionally, {py:class}`CMP<cooper.cmp.ConstrainedMinimizationProblem>`s serve as an interface between the user and **Cooper**, enabling access to {py:meth}`CMP.constraints<ConstrainedMinimizationProblem.constraints>`, {py:meth}`CMP.multipliers<ConstrainedMinimizationProblem.multipliers>`, and {py:meth}`CMP.penalty_coefficients<ConstrainedMinimizationProblem.penalty_coefficients>`.


```{eval-rst}
.. autoclass:: ConstrainedMinimizationProblem
    :members:
```

## CMPStates

A {py:class}`CMPState` is a dataclass containing the information about the loss and constraint violations measured at a specific point. The constraints included in the {py:class}`CMPState` must be passed as a dictionary, where the keys are the {py:class}`Constraint` objects and the values are the associated {py:class}`ConstraintState` objects.


:::{note}
To ensure that your {py:class}`CMPState` is correctly constructed—with loss and constraint tensors that have gradients—you can use the {py:meth}`CMP.sanity_check_cmp_state<ConstrainedMinimizationProblem.sanity_check_cmp_state>` method.

However, avoid calling this method in every iteration of your optimization loop, as it can introduce unnecessary overhead.

:::

```{eval-rst}
.. autoclass:: CMPState
    :members:
```

(cmp-checkpointing)=
## Checkpointing

You can checkpoint a CMP by saving and loading its state with the methods {py:meth}`CMP.state_dict<cooper.ConstrainedMinimizationProblem.state_dict>` and {py:meth}`CMP.load_state_dict<cooper.ConstrainedMinimizationProblem.load_state_dict>`. These methods capture the current values of the multipliers and penalty coefficients for all problem constraints, allowing you to resume the optimization process from that exact state.

```python
# Save the state of the CMP
cmp_state = cmp.state_dict()
torch.save(cmp_state, "cmp_state.pth")

# Later, restore the state of the CMP
cmp = MyCMP(...) # A new CMP with default multiplier and coefficient values
loaded_state = torch.load("cmp_state.pth")
cmp.load_state_dict(loaded_state) # Load checkpointed multipliers and coefficients
```

For a full working example, see [this tutorial](https://cooper.readthedocs.io/en/latest/notebooks/plot_mnist_logistic_regression.html).
