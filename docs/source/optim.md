(optim)=


```{eval-rst}
.. currentmodule:: cooper.optim
```


# Optim

The `cooper.optim` module contains classes and functions for solving constrained minimization problems (CMPs).

This module is divided into three main parts:
- [Constrained Optimizers](#constrained-optimizers): for solving *constrained* minimization problems.
- [Unconstrained Optimizers](#unconstrained-optimizers): for solving *unconstrained* minimization problems.
- [Torch Optimizers](#torch-optimizers): **Cooper** implementations of {py:class}`torch.optim.Optimizer` classes useful for solving CMPs that are not part of PyTorch.

## Quick Start


A {py:class}`~cooper.optim.constrained_optimizers.ConstrainedOptimizer` performs parameter updates to solve a {py:class}`~cooper.ConstrainedMinimizationProblem`. This class wraps two {py:class}`torch.optim.Optimizer` objects: one for the *primal* parameters $\vx$ and one for the *dual* parameters $\vlambda$ and $\vmu$. We refer to these as the primal and dual optimizers, respectively.



:::{admonition} Constrained Optimizers in **Cooper**
:class: note

**Cooper** implements the following subclasses of {py:class}`~cooper.optim.constrained_optimizers.ConstrainedOptimizer`:

- {py:class}`~cooper.optim.constrained_optimizers.SimultaneousOptimizer`: Updates the primal and dual parameters simultaneously.
- {py:class}`~cooper.optim.constrained_optimizers.AlternatingPrimalDualOptimizer`: Performs alternating updates, starting with the primal parameters followed by the dual parameters.
- {py:class}`~cooper.optim.constrained_optimizers.AlternatingDualPrimalOptimizer`: Performs alternating updates, starting with the dual parameters followed by the primal parameters.
- {py:class}`~cooper.optim.constrained_optimizers.ExtrapolationConstrainedOptimizer`: Performs extragradient updates {cite:p}`korpelevich1976extragradient`.

:::

All {py:class}`~cooper.optim.constrained_optimizers.ConstrainedOptimizer`s expect the following arguments:
- `cmp`: a {py:class}`~cooper.ConstrainedMinimizationProblem`.
- `primal_optimizers`: a {py:class}`torch.optim.Optimizer` (or a list of optimizers) for the primal parameters.
- `dual_optimizers`: a {py:class}`torch.optim.Optimizer` (or a list of optimizers) for the dual parameters.


:::{admonition} Unconstrained problems in **Cooper**
:class: note

To accommodate the solution of unconstrained problems using **Cooper**, we provide a {py:class}`~cooper.optim.UnconstrainedOptimizer` class. This is useful for handling both unconstrained problems, as well as formulations of constrained problems without dual variables (e.g., the {py:class}`~cooper.formulations.QuadraticPenalty` formulation). This design allows the use of the `roll()` interface regardless of whether the problem is constrained or unconstrained.

:::

### The `roll()` Method

{py:class}`~cooper.optim.constrained_optimizers.ConstrainedOptimizer` objects define a `roll()` method that prescribes how and when to update the primal and dual parameters. This method is used to perform a single iteration of the optimization algorithm, following PyTorch's `zero_grad() -> forward() -> backward() -> step()` approach.

The `roll()` method is responsible for:
- **Zeroing Gradients**: Calling `primal_optimizer.zero_grad()` and `dual_optimizer.zero_grad()`.
- **Forward Computations**:
  1. Computing the problem's {py:class}`~cooper.CMPState` by calling {py:meth}`cooper.ConstrainedMinimizationProblem.compute_cmp_state()`.
  2. Calculating the primal and dual Lagrangians.
- **Backward** Calling {py:meth}`torch.Tensor.backward()` on the Lagrangian terms.
- **Step**: 
  1. Calling {py:meth}`torch.optim.Optimizer.step()` on the primal and dual optimizers.
  2. Projecting the dual-variables associated with inequality constraints to the non-negative orthant by calling {py:meth}`cooper.multipliers.Multiplier.post_step_()`.

As the procedures for performing updates on the parameters of a CMP can be complex, the `roll()` method provides a convenient and consistent interface for performing parameter updates across {py:class}`~cooper.optim.constrained_optimizers.ConstrainedOptimizer`s. Therefore, when using a {py:class}`~cooper.optim.constrained_optimizers.ConstrainedOptimizer`, users are expected to call the `roll()` method, instead of the individual `step()` methods of the primal and dual optimizers.

```{eval-rst}
.. automethod:: cooper.optim.CooperOptimizer.roll
```

The `roll()` method returns a {py:class}`~cooper.optim.RollOut` object. This includes the computed loss, {py:class}`~cooper.CMPState`, and the primal and dual Lagrangians (packed into {py:class}`~cooper.LagrangianStore` objects). This information can be useful for logging and debugging purposes.

For example, to access the primal Lagrangian you can use the following code snippet:

```python
roll_out = constrained_optimizer.roll(compute_cmp_state_kwargs={...})
primal_lagrangian = roll_out.primal_lagrangian_store.lagrangian
```

```{eval-rst}
.. autoclass:: cooper.optim.RollOut
```

```{eval-rst}
.. autoclass:: cooper.LagrangianStore
```


### Example

To use a {py:class}`~cooper.optim.constrained_optimizers.ConstrainedOptimizer` with a {py:class}`~cooper.formulations.Lagrangian` formulation, follow these steps:

- **\[Line 8\]**: Instantiate a `primal_optimizer` for the primal parameters.
- **\[Line 12\]**: Instantiate a `dual_optimizer` for the dual parameters. Set `maximize=True` since the dual parameters maximize the Lagrangian.
    :::{admonition} Extracting the dual parameters
    :class: tip

    Similar to {py:meth}`torch.nn.Module.parameters()`, {py:class}`~cooper.ConstrainedMinimizationProblem` objects provide a helper method for extracting the dual parameters for all of its associated constraints: {py:meth}`cooper.ConstrainedMinimizationProblem.dual_parameters()`.
    :::
- **\[Lines 16-20\]**: Instantiate a {py:class}`~cooper.optim.constrained_optimizers.ConstrainedOptimizer`, passing the `cmp`, `primal_optimizer`, and `dual_optimizer` as arguments.
- **\[Line 26\]**: Use the `roll` method to perform a *single* call to the `step` method of both the primal and dual optimizers.


```{code-block} python
:emphasize-lines: 8, 12, 16-20, 26
:linenos: true

import torch
import cooper

train_loader = ... # PyTorch DataLoader
model = ... # PyTorch model
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

    constrained_optimizer.roll(compute_cmp_state_kwargs={"model": model, "inputs": inputs, "targets": targets})
```


## Constrained Optimizers

```{eval-rst}
.. currentmodule:: cooper.optim.constrained_optimizers
```


{py:class}`~cooper.optim.constrained_optimizers.ConstrainedOptimizer` objects are used to solve {py:class}`~cooper.ConstrainedMinimizationProblem`s (CMPs). This is achieved via gradient-based optimization of the primal and dual parameters.

:::{admonition} Projected $\vlambda$ Updates
:class: note

To ensure the non-negativity of Lagrange multipliers associated with inequality constraints, all {py:class}`~cooper.optim.constrained_optimizers.ConstrainedOptimizer`s call the {py:meth}`cooper.multipliers.Multiplier.post_step_()` method after dual parameter updates, which projects the multipliers onto the non-negative orthant.

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