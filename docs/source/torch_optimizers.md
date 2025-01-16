(torch_optimizers)=

## Torch Optimizers


```{eval-rst}
.. currentmodule:: cooper.optim.torch_optimizers
```

PyTorch provides implementations of many popular optimizers for solving unconstrained minimization problems. **Cooper** extends PyTorch's functionality by offering optimizers tailored for min-max optimization problems involving Lagrange multipliers, such as the {py:class}`~cooper.formulations.Lagrangian` and {py:class}`~cooper.formulations.AugmentedLagrangian` formulations.

The following optimizers are implemented in **Cooper**:
- {py:class}`~cooper.optim.torch_optimizers.nuPI`: The $\nu$PI optimizer, as introduced by {cite:t}`sohrabi2024nupi`.
- {py:class}`~cooper.optim.torch_optimizers.ExtragradientOptimizer`: A base class for optimizers compatible with {py:class}`~cooper.optim.constrained_optimizers.ExtrapolationConstrainedOptimizer`, **Cooper**'s implementation of the extragradient method.
- Specific instances of the {py:class}`~cooper.optim.torch_optimizers.ExtragradientOptimizer` class, such as the {py:class}`~cooper.optim.torch_optimizers.ExtraSGD` and {py:class}`~cooper.optim.torch_optimizers.ExtraAdam` optimizers.


### $\nu$PI

The $\nu$PI optimizer is a first-order optimization algorithm introduced by {cite:t}`sohrabi2024nupi`. It generalizes several popular first-order optimization techniques, including gradient descent, gradient descent with Polyak momentum {cite:p}`polyak1964some`, Nesterov accelerated gradient {cite:p}`nesterov1983method`, the optimistic gradient method {cite:p}`popov1980modification`, and Proportional-Integral (PI) controllers {cite:p}`astrom1995pid`.


The $\nu$PI optimizer has been shown to reduce oscillations and overshoot in the value of the Lagrange multipliers, leading to more stable convergence to feasible solutions. For a detailed discussion on the $\nu$PI algorithm, see the ICML 2024 paper: [On PI Controllers for Updating Lagrange Multipliers in Constrained Optimization](https://openreview.net/forum?id=1khG2xf1yt).

```{eval-rst}
.. autoenum:: cooper.optim.torch_optimizers.nupi_optimizer.InitType
    :members:
```

```{eval-rst}
.. autoclass:: nuPI
    :members: __init__, step
```

(extragradient-optimizers)=

### Extragradient Optimizers

Extragradient optimizers are PyTorch optimizers equipped with an `extrapolation` method, allowing them to be used alongside the {py:class}`~cooper.optim.constrained_optimizers.ExtrapolationConstrainedOptimizer`.

In **Cooper**, we implement two extragradient optimizers: {py:class}`~cooper.optim.torch_optimizers.ExtraSGD` and {py:class}`~cooper.optim.torch_optimizers.ExtraAdam`. We also provide a base class, {py:class}`~cooper.optim.torch_optimizers.ExtragradientOptimizer`, that can be used to create custom extragradient optimizers.

The implementations of {py:class}`~cooper.optim.torch_optimizers.ExtraSGD` and {py:class}`~cooper.optim.torch_optimizers.ExtraAdam` in **Cooper** are based on minor modifications to the original implementations by [Hugo Berard](https://github.com/GauthierGidel/Variational-Inequality-GAN/blob/master/optim/extragradient.py). For a concise overview of the extra-gradient algorithm and its application to solving Variational Inequality Problems, refer to {cite:p}`gidel2018variational`.

```{eval-rst}
.. autoclass:: ExtragradientOptimizer
    :members:
```

```{eval-rst}
.. autoclass:: ExtraSGD
    :members:
```

```{eval-rst}
.. autoclass:: ExtraAdam
    :members:
```
