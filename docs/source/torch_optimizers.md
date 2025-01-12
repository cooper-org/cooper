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

Credit to original implementations

The implementations of {py:class}`~cooper.optim.ExtraSGD` and
{py:class}`~cooper.optim.ExtraAdam` included in **Cooper** are minor edits from
those originally written by [Hugo Berard](https://github.com/GauthierGidel/Variational-Inequality-GAN/blob/master/optim/extragradient.py).
{cite:t}`gidel2018variational` provides a concise presentation of the
extra-gradient in the context of solving Variational Inequality Problems.

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

TODO: base class
