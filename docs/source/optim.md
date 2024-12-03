(optim)=



# Optim

The optim module contains classes and functions for solving constrained minimization problems (CMPs).

This module is divided into three main parts:
- [Constrained Optimizers](#constrained-optimizers): for solving *constrained* minimization problems.
- [Unconstrained Optimizers](#unconstrained-optimizers): for solving *unconstrained* minimization problems.
- [Torch Optimizers](#torch-optimizers): **Cooper** implementations of useful {py:class}`torch.optim.Optimizer` objects not found in the PyTorch library.

## Quick Start


A {py:class}`~cooper.optim.constrained_optimizers.ConstrainedOptimizer` performs parameter updates to solve a {py:class}`~cooper.ConstrainedMinimizationProblem`.

This class wraps two {py:class}`torch.optim.Optimizer` objects: one for the *primal* parameters $\vx$ and one for the *dual* parameters $\vlambda$ and $\vmu$. Constrained optimizers define procedures for calling the {py:meth}`~torch.optim.Optimizer.step()` method of each optimizer. This procedure is implemented through the {py:meth}`~cooper.optim.CooperOptimizer.roll()` method.

**Cooper** implements the following subclasses of {py:class}`~cooper.optim.constrained_optimizers.ConstrainedOptimizer`:

- {py:class}`~cooper.optim.constrained_optimizers.SimultaneousOptimizer`: Updates the primal and dual parameters simultaneously.
- {py:class}`~cooper.optim.constrained_optimizers.AlternatingPrimalDualOptimizer`: Alternates updates, starting with the primal parameters followed by the dual parameters.
- {py:class}`~cooper.optim.constrained_optimizers.AlternatingDualPrimalOptimizer`: Alternates updates, starting with the dual parameters followed by the primal parameters.
- {py:class}`~cooper.optim.constrained_optimizers.ExtrapolationConstrainedOptimizer`: Utilizes the extragradient method for updates.

All {py:class}`~cooper.optim.constrained_optimizers.ConstrainedOptimizer`s expect the following arguments:
- `cmp`: a {py:class}`~cooper.ConstrainedMinimizationProblem`.
- `primal_optimizers`: a {py:class}`torch.optim.Optimizer` (or a list of optimizers) for the primal parameters.
- `dual_optimizer`: a {py:class}`torch.optim.Optimizer` (or a list of optimizers) for the dual parameters.

:::{admonition} Unconstrained problems in **Cooper**
:class: note

For handling **unconstrained** problems in a consistent way, we provide an
{py:class}`~cooper.optim.UnconstrainedOptimizer` class. {py:class}`~cooper.optim.UnconstrainedOptimizer`s do not expect a `dual_optimizer`.
:::

### Example

To use a {py:class}`~cooper.optim.constrained_optimizers.ConstrainedOptimizer`, follow these steps:
- **\[Line 8\]**: Instantiate a `primal_optimizer` for the primal parameters.
- **\[Line 12\]**: Instantiate a `dual_optimizer` for the dual parameters. Set `maximize=True` since the dual parameters maximize the Lagrangian.
    :::{admonition} Extracting the dual parameters
    :class: tip

    Similar to {py:meth}`torch.nn.Module.parameters()`, {py:class}`~cooper.ConstrainedMinimizationProblem` objects provide a helper method for extracting the dual parameters for all of its associated constraints: {py:meth}`cooper.ConstrainedMinimizationProblem.dual_parameters()`.
    :::
- **\[Lines 17-21\]**: Instantiate a {py:class}`~cooper.optim.constrained_optimizers.ConstrainedOptimizer`, passing the `cmp`, `primal_optimizer`, and `dual_optimizer` as arguments.
- **\[Line 28\]**: Use the `roll` method to perform a *single* call to the `step` method of both the primal and dual optimizers.


```{code-block} python
:emphasize-lines: 8, 12, 17-21, 28
:linenos: true

import torch
import cooper

train_loader = ...
model = ... # a PyTorch model
cmp = ... # containing `Constraint`s and their associated `Multiplier`s

primal_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# `cmp.dual_parameters()` returns the parameters associated with the multipliers.
# Must set `maximize=True` since the multipliers *maximize* the Lagrangian.
dual_optimizer = torch.optim.SGD(cmp.dual_parameters(), lr=1e-3, maximize=True)

# `ConstrainedOptimizer`s need access to the cmp to compute the loss, constraints, and
# Lagrangian. Some `ConstrainedOptimizer`s do these calculations multiple times.
constrained_optimizer = cooper.SimultaneousOptimizer(
    cmp=cmp,
    primal_optimizers=primal_optimizer,
    dual_optimizer=dual_optimizer,
)

for inputs, targets in train_loader:
    # kwargs used by `cmp.compute_cmp_state` method to compute the loss and constraints.
    kwargs = {"model": model, "inputs": inputs, "targets": targets}

    # roll is a convenience method that
    constrained_optimizer.roll(compute_cmp_state_kwargs={"model": model, "inputs": inputs, "targets": targets})
```


## Constrained Optimizers

```{eval-rst}
.. currentmodule:: cooper.optim.constrained_optimizers
```


{py:class}`~cooper.optim.constrained_optimizers.ConstrainedOptimizer` objects are used to solve {py:class}`~cooper.ConstrainedMinimizationProblem`s (CMPs). This is achieved via gradient-based optimization of the primal and dual parameters.

:::{admonition} Projected $\vlambda$ Updates
:class: note

To ensure the non-negativity of Lagrange multipliers associated with inequality constraints, all {py:class}`~cooper.optim.constrained_optimizers.ConstrainedOptimizer`s call the {py:meth}`cooper.multipliers.Multiplier.post_step()` method after dual parameter updates, which projects the multipliers onto the non-negative orthant.

:::


### Simultaneous Optimizer


```{eval-rst}
.. autoclass:: SimultaneousOptimizer
    :members:
```


### Alternating Optimizers

Point about efficiency:
- PrimalDual: use compute_violations

```{eval-rst}
.. autoclass:: AlternatingPrimalDualOptimizer
    :members:
```

```{eval-rst}
.. autoclass:: AlternatingDualPrimalOptimizer
    :members:
```


### Extragradient

```{eval-rst}
.. autoclass:: ExtrapolationConstrainedOptimizer
    :members:
```



### Base Class

How to implement a custom optimizer

Description of the zero_grad -> forward -> Lagrangian -> backward -> step -> projection


```{eval-rst}
.. autoclass:: cooper.optim.constrained_optimizers.ConstrainedOptimizer
    :members:
```




## **Cooper** Optimizer Base Class

```{eval-rst}
.. currentmodule:: cooper.optim
```

```{eval-rst}
.. autoclass:: CooperOptimizer
    :members:
```


## Unconstrained Optimizers


```{eval-rst}
.. autoclass:: UnconstrainedOptimizer
    :members:
```


## Torch Optimizers

```{eval-rst}
.. currentmodule:: cooper.optim.torch_optimizers
```


### nuPI

```{eval-rst}
.. autoclass:: nuPI
    :members:
```


### Extragradient Optimizers

Credit to original implementations

```{eval-rst}
.. autoclass:: ExtraSGD
    :members:
```

```{eval-rst}
.. autoclass:: ExtraAdam
    :members:
```