(multipliers)=

# Multipliers

```{eval-rst}
.. currentmodule:: cooper.multipliers
```

Multiplier objects represent the dual variables $ \vlambda $ and $ \vmu $ in the constrained problem. They are required by certain formulations, such as {py:class}`~cooper.formulations.Lagrangian` and {py:class}`~cooper.formulations.AugmentedLagrangian`.

In a generic formulation $ \Lag $, the dual variables correspond to the inner maximization variables:

$$
\min_{\vx \in \reals^d} \,\, \max_{\vlambda \ge \vzero, \vmu} \,\, \Lag(\vx,\vlambda, \vmu).
$$

In **Cooper**, multipliers are implemented as {py:class}`torch.nn.Module`s, ensuring compatibility with PyTorch's autograd capabilities. They are evaluated via a {py:meth}`~cooper.multipliers.Multiplier.forward` call.

The `cooper.multipliers` module provides three types of multipliers:

- **{py:class}`~cooper.multipliers.DenseMultiplier`**: Represents each multiplier individually, with each entry in the multiplier vector corresponding to a separate constraint.
- **{py:class}`~cooper.multipliers.IndexedMultiplier`**: Similar to {py:class}`~cooper.multipliers.DenseMultiplier`, but supports efficient indexing. This is useful for scenarios where constraints are sampled, allowing for sparse multiplier accessing and updates (see [Constraint Sampling](problem.md#constraint-sampling)).
- **{py:class}`~cooper.multipliers.ImplicitMultiplier`**: Instead of storing multipliers explicitly, {py:class}`~cooper.multipliers.ImplicitMultiplier`s compute their values through a `forward` call on an *arbitrary* {py:class}`torch.nn.Module`. This is particularly useful when the number of constraints is too large to maintain individual multipliers.

The diagram below illustrates how different multiplier types operate. Intuitively, all multipliers can be viewed as modules that produce vectors of multiplier values. These values must align with the constraint violations stored in the `violation` tensor of a {py:class}`~cooper.constraints.ConstraintState`. The color coding in the diagram indicates this alignment between individual constraint violations and their corresponding multiplier values.

TODO: diagram
![multipliers](_static/multipliers.svg)

% Duplicating "Linking constraints and multipliers" on problem.md

:::{admonition} Linking constraints and multipliers
:class: hint

{py:class}`~cooper.constraints.Constraint` objects require an associated {py:class}`~cooper.multipliers.Multiplier` when the problem formulation demands it. You can check this using the {py:attr}`~cooper.formulations.Formulation.expects_multiplier` attribute of a {py:class}`~cooper.formulations.Formulation` sub-class. To ensure compliance, pass a {py:class}`~cooper.multipliers.Multiplier` object to the {py:class}`~cooper.constraints.Constraint` constructor.

```python
multiplier = ...
constraint = cooper.Constraint(
    multiplier=multiplier,
    constraint_type=cooper.ConstraintType.INEQUALITY,
    formulation_type=cooper.formulations.Lagrangian,
)
```
:::


:::{note}

The helper methods {py:meth}`CMP.multipliers<.ConstrainedMinimizationProblem.multipliers>` and {py:meth}`CMP.named_multipliers<.ConstrainedMinimizationProblem.named_multipliers>` allow iteration over the multipliers associated with constraints registered in a {py:class}`CMP<cooper.cmp.ConstrainedMinimizationProblem>`.
For more details, see [Registering constraints in a CMP](#registering-constraints).
:::


## Explicit (Non-Parametric) Multipliers

Consider the following Lagrangian formulation of a constrained optimization problem:

$$
\Lag(\vx, \vlambda, \vmu) = f(\vx) + \vlambda^\top \vg(\vx) + \vmu^\top \vh(\vx),
$$

where $\vlambda = [\lambda_i]_{i=1}^m$ and $\vmu = [\mu_i]_{i=1}^n$ are the Lagrange multipliers associated with the equality and inequality constraints, respectively.

{py:class}`~cooper.multipliers.ExplicitMultiplier` objects represent the vectors $\vlambda$ and $\vmu$ directly, by storing *one decision variable per constraint*.
**Cooper** provides two types of explicit multipliers: {py:class}`~cooper.multipliers.DenseMultiplier`s and {py:class}`~cooper.multipliers.IndexedMultiplier`s.

### Initialization

To create an {py:class}`~cooper.multipliers.ExplicitMultiplier`, you must specify either (i) the number of associated constraints or (ii) an initial value for each multiplier entry.

The example below illustrates how to construct multiplier objects. Note that the syntax is consistent between {py:class}`~cooper.multipliers.DenseMultiplier` and {py:class}`~cooper.multipliers.IndexedMultiplier`.

```python
# When specifying the number of constraints, all multipliers are initialized to zero
multiplier = cooper.multipliers.DenseMultiplier(
    num_constraints=3,
    device=torch.device("cpu"),
    dtype=torch.float32,
)

# When `init` is provided, `num_constraints` is inferred from its shape
multiplier = cooper.multipliers.IndexedMultiplier(
    init=torch.ones(7),
    device=torch.device("cuda"),
    dtype=torch.float16,
)
```

### Evaluating an {py:class}`~cooper.multipliers.ExplicitMultiplier`

**Cooper** stores multiplier values in the `weight` attribute, which can be accessed for inspecting their behavior.

However, to leverage PyTorch's autograd functionality, we recommend evaluating multipliers via their {py:meth}`~cooper.multipliers.ExplicitMultiplier.forward` method. For example:

```python
# `DenseMultiplier`s do not require arguments during evaluation
multiplier_value = multiplier()

# `IndexedMultiplier`s require indices for evaluation
indices = torch.tensor([0, 2, 4, 6])
multiplier_value = multiplier(indices)
```

```{eval-rst}
.. autoclass:: ExplicitMultiplier
    :members: initialize_weight, sanity_check, post_step_, forward
```

### Dense Multipliers

{py:class}`~cooper.multipliers.DenseMultiplier` objects are ideal for problems with a small to medium number of constraints, where all constraints are measured or observed at each iteration.

We refer to this type of multiplier as *dense* because all multipliers are utilized at every optimization step (e.g., during the computation of the Lagrangian), as opposed to only a subset being used.

A {py:class}`~cooper.multipliers.DenseMultiplier` is essentially a wrapper around a {py:class}`torch.Tensor` to provide an interface consistent with other types of multipliers.
It implements the {py:meth}`~cooper.multipliers.DenseMultiplier.forward` method, which returns all multipliers as a single tensor. This method takes no arguments.

For large-scale {py:class}`~cooper.constraints.Constraint` objects (e.g., one constraint per training example) or problems where constraints are sampled and not all constraints are observed simultaneously, consider using an {py:class}`~cooper.multipliers.IndexedMultiplier` or {py:class}`~cooper.multipliers.ImplicitMultiplier` instead.


```{eval-rst}
.. autoclass:: DenseMultiplier
    :members:
```


### Indexed Multipliers

Like {py:class}`~cooper.multipliers.DenseMultiplier`s, {py:class}`~cooper.multipliers.IndexedMultiplier`s represent the multiplier tensors directly, but allow efficiently accessing and updating specific entries *by index*.

{py:class}`~cooper.multipliers.IndexedMultiplier` objects are designed for situations where only a subset of constraints are observed at each iteration, rather than all constraints. This approach is particularly useful when the number of constraints is large, such as in tasks where a constraint is imposed for each data. In such cases, measuring all constraints simultaneously can be computationally prohibitive.

{py:class}`~cooper.multipliers.IndexedMultiplier` objects model $\vlambda$ and $\vmu$ explicitly, just like {py:class}`~cooper.multipliers.DenseMultiplier`s, but allow fetching and updating them *by index*. Given indices `idx`, the {py:meth}`~cooper.multipliers.IndexedMultiplier.forward()` method of an {py:class}`~cooper.multipliers.IndexedMultiplier` object returns the multipliers corresponding to the indices in `idx`. {py:class}`~cooper.multipliers.IndexedMultiplier`s enable time-efficient retrieval of multipliers for only the sampled constraints, while also supporting memory-efficient sparse gradients (on GPU).

To use an {py:class}`~cooper.multipliers.IndexedMultiplier`, after computing the constraints you must provide the observed constraint indices to the `constraint_features` argument of the {py:meth}`~cooper.constraints.ConstraintState`. **Cooper** will then know which multipliers to fetch and update during optimization. For example, if you measured the constraints at indices 0, 11, and 17, you would set the `constraint_features` attribute as follows:

```python
observed_violation_tensor = torch.tensor([3, 1, 4])
observed_constraint_indices = torch.tensor([0, 11, 17])

constraint_state = cooper.ConstraintState(
    violation=observed_violation_tensor,
    constraint_features=observed_constraint_indices
)
```

```{warning}
The {py:meth}`~cooper.multipliers.IndexedMultiplier.forward` call of an {py:class}`~cooper.multipliers.IndexedMultiplier` *expects* a list of indices corresponding to the constraints whose multipliers you want to fetch. If you want to fetch all multipliers, you must provide a list of *all* constraint indices.
```

```{eval-rst}
.. autoclass:: IndexedMultiplier
    :members:
```


## Implicit (Parametric) Multipliers

Rather than maintaining a separate learnable parameter for each multiplier, it can be more practical to model the multipliers as *functions* of the constraints. This approach is particularly beneficial when the number of constraints is large, as explicitly maintaining a Lagrange multiplier for each constraint can be computationally expensive or infeasible.

We can represent the multipliers functionally as follows: $\vlambda_{\phi}: \mathbb{R}^d \to \mathbb{R}^m$ and $\vmu_{\psi}: \mathbb{R}^d \to \mathbb{R}^n$. In this representation, $d$ refers to the dimensionality of the constraint feature space, while $\phi$ and $\psi$ are the parameters of the multiplier functions. These functions calculate the multiplier values based on the input constraint features.

To support this functional approach, **Cooper** provides the {py:class}`~cooper.multipliers.ImplicitMultiplier` class. To use it, you must implement its `forward()` method, which takes a tensor of constraint features as input and returns the corresponding multiplier values. For example, to define a linear model for the multipliers, you can write:

```python
import torch
import torch.nn as nn

class ImplicitMultiplier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, constraint_features):
        return self.linear(constraint_features)
```

This approach can be extended to more complex models, such as neural networks.

Much like the {py:class}`~cooper.multipliers.IndexedMultiplier`, you need to provide the constraint features associated with the observed constraints to the `constraint_features` attribute of the {py:class}`~cooper.constraints.ConstraintState`. **Cooper** will then perform forward and backward passes through the multiplier model and update its parameters accordingly.

:::{note}
Due to their functional nature, implicit multipliers can (approximately) represent _infinitely_ many constraints. This capability is inspired by the "Lagrange multiplier model" proposed by {cite:p}`narasimhan2020multiplier`.
:::

:::{warning}
Because of the high flexibility of implicit multipliers, the `post_step_` method is not implemented in the base class. For applications involving inequality constraints and implicit multipliers, you must implement the logic to ensure the non-negativity of the multipliers associated with inequality constraints.
:::


```{eval-rst}
.. autoclass:: ImplicitMultiplier
    :members:
```

## Base Class

```{eval-rst}
.. autoclass:: Multiplier
    :members:
```
