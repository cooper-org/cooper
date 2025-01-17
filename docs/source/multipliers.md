(multipliers)=

# Multipliers

```{eval-rst}
.. currentmodule:: cooper.multipliers
```

Multiplier objects represent the dual variables of the optimization problem: :math:`\vlambda` and :math:`\vnu`.

## Example

[Dense Multipliers](#Dense-Multipliers) can be initialized in one of the following ways:

```python
import cooper

multiplier = cooper.DenseMultiplier(num_constraints=3, device=torch.device("cuda"))
```

```python
import cooper

multiplier = cooper.DenseMultiplier(init=torch.ones(3), device=torch.device("cuda"))
```

## Constructing a DenseMultiplier

The construction of a `DenseMultiplier` requires its initial value `init` to
be provided. The shape of `init` should match that shape of the corresponding
constraint defect. Recall that in {ref}`lagrangian_formulations`, the
calculation of the Lagrangian involves an inner product between the multipliers
and the constraint violations.

We lazily initialize `DenseMultiplier`s via a call to the
{py:meth}`~cooper.formulation.lagrangian.BaseLagrangianFormulation.create_state`
method of a {py:class}`~cooper.formulation.lagrangian.BaseLagrangianFormulation`
object. This happens when the first `CMPState` of the
`ConstrainedMinimizationProblem` is evaluated. That way we can ensure that the
`DenseMultiplier`s are initialized on the correct device and that their shape
matches that of the constraint defects. If initial values for multipliers are
not specified during the initialization of the `BaseLagrangianFormulation`,
they are initialized to zero.

After having performed a step on the dual variables inside the
{py:meth}`~cooper.optim.constrained_optimizers.ConstrainedOptimizer.step` method of
{py:class}`~cooper.optim.constrained_optimizers.ConstrainedOptimizer`, a call to
{py:meth}`~DenseMultiplier.project_` ensures that the multiplier values are
admissible. It is possible to override the {py:meth}`~DenseMultiplier.project_`
method in order to apply a custom projection to the multipliers.

In the case of a {py:class}`~cooper.formulation.lagrangian.LagrangianFormulation`,
the projection only affects the Lagrange multipliers associated with the inequality
constraints.

The flag `positive` denotes whether a multiplier corresponds to an inequality
or equality constraint, and thus whether the multiplier value must be
lower-bounded by zero or not. `positive=True` corresponds to inequality
constraints, while `positive=False` corresponds to equality constraints.





## Base Class

```{eval-rst}
.. autoclass:: Multiplier
    :members:
```

## Explicit (Non-Parametric) Multipliers

```{eval-rst}
.. autoclass:: ExplicitMultiplier
    :members:
```

### Dense Multipliers

```{eval-rst}
.. autoclass:: DenseMultiplier
    :members:
```


### Indexed Multipliers

```{eval-rst}
.. autoclass:: IndexedMultiplier
    :members:
```


## Implicit (Parametric) Multipliers

Certain optimization problems involve a very large number of constraints. For
example, in a learning task one might impose a constraint *per data-point*.
In these settings, explicitly maintaining one Lagrange multiplier per constraint
becomes impractical {cite:p}`narasimhan2020multiplier`. We provide a
`BaseMultiplier` class which can be easily extended to accommodate situations
which employ sparse multipliers or even a model that predicts the value of the
multiplier based on some properties or "features" of each constraint.

```{eval-rst}
.. autoclass:: ImplicitMultiplier
    :members:
```
