(multipliers)=

# Multipliers

```{eval-rst}
.. currentmodule:: cooper.multipliers
```

Multiplier objects represent the dual variables $\vlambda$ and $\vmu$ of the constrained optimization problem. They are required by certain formulations, such as the {py:class}`~cooper.formulations.Lagrangian` and {py:class}`~cooper.formulations.AugmentedLagrangian` formulations.

In **Cooper**, multipliers inherit from {py:class}`torch.nn.Module`, ensuring compatibility with PyTorch's autograd capabilities. In particular, multipliers are evaluated using a {py:meth}`~cooper.multipliers.Multiplier.forward` method.

This module provides the following main classes:
- **{py:class}`~cooper.multipliers.DenseMultiplier`**: Models each multiplier individually.
- **{py:class}`~cooper.multipliers.IndexedMultiplier`**: Similar to `DenseMultiplier` but allows fetching and updating multipliers by index. Useful when constraints are sampled, and thus the required multipliers change at each iteration.
- **{py:class}`~cooper.multipliers.ImplicitMultiplier`**: Models multipliers implicitly as a function of some features of the constraints. Suitable when the number of constraints is very large, making it impractical or impossible to explicitly maintain a Lagrange multiplier for each constraint.


## Explicit (Non-Parametric) Multipliers

The {py:class}`~cooper.multipliers.ExplicitMultiplier` class considers *one learnable parameter per constraint*. Consider the following Lagrangian formulation of a constrained optimization problem:

$$
\Lag(\vx, \vlambda, \vmu) = f(\vx) + \vlambda^\top \vg(\vx) + \vmu^\top \vh(\vx),
$$

where $\vlambda = [\lambda_i]_{i=1}^m$ and $\vmu = [\mu_i]_{i=1}^n$ are the Lagrange multipliers associated with the equality and inequality constraints, respectively. {py:class}`~cooper.multipliers.ExplicitMultiplier` objects represent the vectors $\vlambda$ and $\vmu$ directly.

We provide two types of explicit multipliers: {py:class}`~cooper.multipliers.DenseMultiplier` and {py:class}`~cooper.multipliers.IndexedMultiplier`.

To initialize an {py:class}`~cooper.multipliers.ExplicitMultiplier`, you need to specify either the number of associated constraints or provide an initial value for each multiplier. For example:

```python
multiplier = cooper.multipliers.DenseMultiplier(
    num_constraints=3,
    device=torch.device("cuda")
)
```

or

```python
multiplier = cooper.multipliers.DenseMultiplier(
    init=torch.ones(3),
    device=torch.device("cuda"),
)
```

When specifying the number of constraints, all multipliers are initialized to zero. If an initial value is provided, the number of constraints is inferred from the length of the provided tensor.

To evaluate an {py:class}`~cooper.multipliers.ExplicitMultiplier` while preserving the computational graph, you can use its {py:meth}`~cooper.multipliers.ExplicitMultiplier.forward` method.


```{eval-rst}
.. autoclass:: ExplicitMultiplier
    :members: initialize_weight, sanity_check, post_step_, forward
```

### Dense Multipliers

{py:class}`~cooper.multipliers.DenseMultiplier`s implement a `forward()` method that expects no arguments and returns all multipliers as a single tensor.  {py:class}`~cooper.multipliers.DenseMultiplier` objects are suitable for problems with a small to medium number of constraints, where all constraints in the group are measured at each iteration.

For large-scale {py:class}`~cooper.constraints.Constraint` objects (e.g., one constraint per training example), or problems where constraints are sampled and not all are observed at once, consider using an {py:class}`~cooper.multipliers.IndexedMultiplier` instead.


```{eval-rst}
.. autoclass:: DenseMultiplier
    :members:
```


### Indexed Multipliers



{py:class}`~cooper.multipliers.IndexedMultiplier` objects model $\vlambda$ and $\vmu$ explicitly, just like {py:class}`~cooper.multipliers.DenseMultiplier`s, but allow fetching and updating them *by index*. Given indices `idx`, the {py:meth}`~cooper.multipliers.IndexedMultiplier.forward()` method of an {py:class}`~cooper.multipliers.IndexedMultiplier` object returns the multipliers corresponding to the indices in `idx`. {py:class}`~cooper.multipliers.IndexedMultiplier`s enable time-efficient retrieval of the multipliers for the sampled constraints only, and memory-efficient sparse gradients (on GPU).

{py:class}`~cooper.multipliers.IndexedMultiplier` objects are designed for situations where only a subset of constraints are observed at each iteration, rather than all constraints. This approach is especially useful when the number of constraints is large, such as in tasks where a constraint is imposed for each data point. In these cases, measuring all constraints at once can be computationally prohibitive.

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
