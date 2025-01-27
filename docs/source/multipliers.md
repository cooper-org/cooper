(multipliers)=

# Multipliers

```{eval-rst}
.. currentmodule:: cooper.multipliers
```

Multiplier objects represent the dual variables $\vlambda$ and $\vmu$ of the constrained optimization problem. They are required by certain formulations, such as the {py:class}`~cooper.formulations.Lagrangian` and {py:class}`~cooper.formulations.AugmentedLagrangian` formulations.

For a generic formulation $\Lag$ the dual variables correspond to the inner maximization:

$$
\min_{\vx \in \reals^d} \,\, \max_{\vlambda \ge \vzero, \vmu} \,\, \Lag(\vx,\vlambda, \vmu).
$$

In **Cooper**, multipliers are {py:class}`torch.nn.Module`s, ensuring compatibility with PyTorch's autograd capabilities. In particular, multipliers are evaluated using a {py:meth}`~cooper.multipliers.Multiplier.forward` call.

The `cooper.multipliers` module provides the following types of multipliers:
- **{py:class}`~cooper.multipliers.DenseMultiplier`**: Represents each multiplier individually, such that each entry of the multiplier vector corresponds to a separate constraint.
- **{py:class}`~cooper.multipliers.IndexedMultiplier`**: Like in {py:class}`~cooper.multipliers.DenseMultiplier`, each entry of the multiplier vector corresponds to a separate constraint. However, {py:class}`~cooper.multipliers.IndexedMultiplier`s support efficient indexing of the Multipliers. This is useful when constraints are sampled (see [Constraint Sampling](problem.md#constraint-sampling)).
- **{py:class}`~cooper.multipliers.ImplicitMultiplier`**: Rather than storing each multiplier explicitly as entries of a vector, {py:class}`~cooper.multipliers.ImplicitMultiplier`s represent the value of the multipliers as the result of a `forward` call on an *arbitrary* {py:class}`torch.nn.Module`. This is suitable when the number of constraints is very large, making it impractical or impossible to explicitly maintain a Lagrange multiplier for each constraint.

The diagram below illustrates the operation of the different multipliers types. Intuitively, the different types of multipliers can be abstracted into modules producing vectors of multiplier values. These values must be aligned with constraint violations provided in the `violation` tensor of a {py:class}`~cooper.constraints.ConstraintState`. This alignment between individual constraint violations and multiplier values is indicated by the color coding.

![multipliers](_static/multipliers.svg)

## Explicit (Non-Parametric) Multipliers

Consider the following Lagrangian formulation of a constrained optimization problem:

$$
\Lag(\vx, \vlambda, \vmu) = f(\vx) + \vlambda^\top \vg(\vx) + \vmu^\top \vh(\vx),
$$

where $\vlambda = [\lambda_i]_{i=1}^m$ and $\vmu = [\mu_i]_{i=1}^n$ are the Lagrange multipliers associated with the equality and inequality constraints, respectively.

{py:class}`~cooper.multipliers.ExplicitMultiplier` objects represent the vectors $\vlambda$ and $\vmu$ directly, by storing *one decision variable per constraint*.
**Cooper** provides two types of explicit multipliers: {py:class}`~cooper.multipliers.DenseMultiplier` and {py:class}`~cooper.multipliers.IndexedMultiplier`.

### Initialization

To create an {py:class}`~cooper.multipliers.ExplicitMultiplier`, you need to specify either the number of associated constraints or provide an initial value for each multiplier entry. We illustrate the construction of multiplier objects below. Note that the syntax is consistent between {py:class}`~cooper.multipliers.DenseMultiplier` and {py:class}`~cooper.multipliers.IndexedMultiplier`.

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

**Cooper** stores the values of the multipliers in the `weight` attribute.
Accessing the `weight` tensor directly can be useful for inspecting the multipliers to ensure proper behavior.


However, to take advantage of the autograd functionality from PyTorch, we recommend evaluating the multipliers via their {py:meth}`~cooper.multipliers.ExplicitMultiplier.forward` method. For example:
```python
# `DenseMultiplier`s do not require arguments during evaluation
multiplier_value = multiplier()

# `IndexedMultiplier`s require indices for evaluation
indices = torch.tensor([0, 2, 4, 6])
multiplier_value = multiplier(indices)
```

### Using a multiplier with a {py:class}`~cooper.constraints.Constraint` and {py:class}`~cooper.cmp.ConstrainedMinimizationProblem`

{py:class}`~cooper.constraints.Constraint`s in **Cooper** must be associated with a {py:class}`~cooper.multipliers.Multiplier` object. To achieve this, provide a {py:class}`~cooper.multipliers.Multiplier` to the {py:class}`~cooper.constraints.Constraint` constructor.
For instance:

```python
constraint = cooper.Constraint(
    multiplier=multiplier,
    constraint_type=cooper.ConstraintType.INEQUALITY,
)
```

{py:class}`~cooper.cmp.ConstrainedMinimizationProblem` objects provide helper functions to iterate over all the multipliers associated with their constraints. See {py:func}`~cooper.cmp.ConstrainedMinimizationProblem.multipliers` for more details.

[Constrained Minimization Problems](problem.md#constrained-minimization-problems) for more details.

```{eval-rst}
.. autoclass:: ExplicitMultiplier
    :members: initialize_weight, sanity_check, post_step_, forward
```

### Dense Multipliers

{py:class}`~cooper.multipliers.DenseMultiplier` objects are suitable for problems with a small to medium number of constraints, where all constraints are measured/observed at each iteration.
We name this type of multiplier *dense* as all multipliers are used at every optimization step (e.g. in the computation of the Lagrangian), rather than just a subset thereof.

{py:class}`~cooper.multipliers.DenseMultiplier` is effectively a wrapper around a {py:class}`torch.Tensor` object to provide an interface that is consistent with other types of multipliers.
{py:class}`~cooper.multipliers.DenseMultiplier`s implement a {py:meth}`~cooper.multipliers.DenseMultiplier.forward` method that returns all multipliers as a single tensor. This method expects no arguments.

For large-scale {py:class}`~cooper.constraints.Constraint` objects (e.g., one constraint per training example), or problems where constraints are sampled and not all are observed at once, consider using an {py:class}`~cooper.multipliers.IndexedMultiplier` or {py:class}`~cooper.multipliers.ImplicitMultiplier` instead.


```{eval-rst}
.. autoclass:: DenseMultiplier
    :members:
```


### Indexed Multipliers

Like {py:class}`~cooper.multipliers.DenseMultiplier`s, {py:class}`~cooper.multipliers.IndexedMultiplier`s represent the multiplier tensors directly, but allow efficiently accessing and updating specific entries *by index*.

{py:class}`~cooper.multipliers.IndexedMultiplier` objects are designed for situations where only a subset of constraints are observed at each iteration, rather than all constraints. This approach is especially useful when the number of constraints is large, such as in tasks where a constraint is imposed for each data point. In these cases, measuring all constraints at once can be computationally prohibitive.



{py:class}`~cooper.multipliers.IndexedMultiplier` objects model $\vlambda$ and $\vmu$ explicitly, just like {py:class}`~cooper.multipliers.DenseMultiplier`s, but allow fetching and updating them *by index*. Given indices `idx`, the {py:meth}`~cooper.multipliers.IndexedMultiplier.forward()` method of an {py:class}`~cooper.multipliers.IndexedMultiplier` object returns the multipliers corresponding to the indices in `idx`. {py:class}`~cooper.multipliers.IndexedMultiplier`s enable time-efficient retrieval of the multipliers for the sampled constraints only, and memory-efficient sparse gradients (on GPU).


To use an {py:class}`~cooper.multipliers.IndexedMultiplier`, after computing the constraints you must provide the observed constraint indices to the `constraint_features` argument of the {py:meth}`~cooper.constraints.ConstraintState`. **Cooper** will then know which multipliers to fetch and update during optimization. For example, if you measured the constraints at indices 0, 11, and 17, you would set the `constraint_features` attribute as follows:

```python
observed_constraint_tensor = torch.tensor([0, 2, 4])
observed_constraint_indices = torch.tensor([0, 11, 17])

constraint_state = cooper.ConstraintState(
    violation=observed_constraint_tensor,
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

Instead of having a separate learnable parameter for each multiplier, it is sometimes more practical to model the multipliers as *functions* of the constraints. This approach is especially useful when dealing with a large number of constraints, where explicitly maintaining a Lagrange multiplier for each constraint becomes impractical or infeasible.

Consider a functional representation of the multipliers: $\vlambda_{\phi}: \mathbb{R}^d \to \mathbb{R}^m$ and $\vmu_{\psi}: \mathbb{R}^d \to \mathbb{R}^n$. Here, $d$ denotes the dimensionality of the feature-space representation for the constraints, while $\phi$ and $\psi$ are the parameters of the multiplier functions. These functions compute the multiplier values based on the input constraint features.

**Cooper** provides the {py:class}`~cooper.multipliers.ImplicitMultiplier` class to facilitate this implicit modeling of multipliers. To use it, you must implement its `forward()` method, which takes a tensor of constraint features as input and outputs the corresponding multiplier values. For instance, to define a linear model for the multipliers, you can write:

```python

from cooper.multipliers import ImplicitMultiplier

class LinearMultiplier(ImplicitMultiplier):
    def __init__(self, num_constraints, feature_dim, device):
        super().__init__(num_constraints, feature_dim, device)
        self.linear = torch.nn.Linear(feature_dim, num_constraints)

    def forward(self, x):
        return self.linear(x)

```
Note that this approach can be extended to more complex models, such as neural networks.

Similar to {py:class}`~cooper.multipliers.IndexedMultiplier`, you must provide the constraint features associated with the observed constraints to the `constraint_features` attribute of the {py:class}`~cooper.constraints.ConstraintState`. **Cooper** will then compute forward and backward passes through the multiplier model and update its parameters.

:::{note}
Thanks to their functional nature, implicit multipliers can (approximately) represent _infinitely_ many constraints. This feature is inspired by the "Lagrange multiplier model" proposed by {cite:p}`narasimhan2020multiplier`.
:::

:::{warning}

Given the high flexibility of implicit multipliers, the `post_step_` method is not implemented in the base class. For applications involving inequality constraints and implicit multipliers, you must implement the logic to maintain the non-negativity of the multipliers associated with inequality constraints.
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
