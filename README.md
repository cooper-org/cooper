# Cooper

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/cooper-org/cooper/tree/master/LICENSE)
[![DOCS](https://readthedocs.org/projects/cooper/badge/?version=latest)](https://cooper.readthedocs.io/en/latest/?version=latest)
[![Build and Test](https://github.com/cooper-org/cooper/actions/workflows/build.yml/badge.svg)](https://github.com/cooper-org/cooper/actions/workflows/build.yml)
[![Codecov](https://codecov.io/gh/cooper-org/cooper/branch/dev/graph/badge.svg?token=1AKM2EQ7RT)](https://codecov.io/gh/cooper-org/cooper/branch/dev/graph/badge.svg?token=1AKM2EQ7RT)

## About

**Cooper** is a toolkit for Lagrangian-based constrained optimization in Pytorch.
This library aims to encourage and facilitate the study of constrained
optimization problems in machine learning.

**Cooper** is (almost!) seamlessly integrated with Pytorch and preserves the
usual `loss -> backward -> step` workflow. If you are already familiar with
Pytorch, using **Cooper** will be a breeze! 🙂

**Cooper** was born out of the need to handle constrained optimization problems
for which the loss or constraints are not necessarily "nicely behaved"
or "theoretically tractable", e.g. when no (efficient) projection or proximal
are available. Although assumptions of this kind have enabled the development of
great Pytorch-based libraries such as [CHOP](https://github.com/openopt/chop)
and [GeoTorch](https://github.com/Lezcano/geotorch), they are seldom satisfied
in the context of many modern machine learning problems.

Many of the structural design ideas behind **Cooper** are heavily inspired by
the [TensorFlow Constrained Optimization (TFCO)](https://github.com/google-research/tensorflow_constrained_optimization)
library. We highly recommend TFCO for TensorFlow-based projects and will
continue to integrate more of TFCO's features in future releases.

⚠️ This library is under active development. Future API changes might break backward
compatibility. ⚠️

## Getting Started

Here we consider a simple convex optimization problem to illustrate how to use
**Cooper**. This example is inspired by [this StackExchange question](https://datascience.stackexchange.com/questions/107366/how-do-you-solve-strictly-constrained-optimization-problems-with-pytorch):

> _I am trying to solve the following problem using Pytorch: given a 6-sided die
> whose average roll is known to be 4.5, what is the maximum entropy
> distribution for the faces?_

```python
import torch
import cooper

class MaximumEntropy(cooper.ConstrainedMinimizationProblem):
    def __init__(self, mean_constraint):
        self.mean_constraint = mean_constraint
        super().__init__()

    def closure(self, probs):
        # Verify domain of definition of the functions
        assert torch.all(probs >= 0)

        # Negative signed removed since we want to *maximize* the entropy
        entropy = torch.sum(probs * torch.log(probs))

        # Entries of p >= 0 (equiv. -p <= 0)
        ineq_defect = -probs

        # Equality constraints for proper normalization and mean constraint
        mean = torch.sum(torch.tensor(range(1, len(probs) + 1)) * probs)
        eq_defect = torch.stack([torch.sum(probs) - 1, mean - self.mean_constraint])

        return cooper.CMPState(loss=entropy, eq_defect=eq_defect, ineq_defect=ineq_defect)

# Define the problem and formulation
cmp = MaximumEntropy(mean_constraint=4.5)
formulation = cooper.LagrangianFormulation(cmp)

# Define the primal parameters and optimizer
probs = torch.nn.Parameter(torch.rand(6)) # Use a 6-sided die
primal_optimizer = cooper.optim.ExtraSGD([probs], lr=3e-2, momentum=0.7)

# Define the dual optimizer. Note that this optimizer has NOT been fully instantiated
# yet. Cooper takes care of this, once it has initialized the formulation state.
dual_optimizer = cooper.optim.partial_optimizer(cooper.optim.ExtraSGD, lr=9e-3, momentum=0.7)

# Wrap the formulation and both optimizers inside a ConstrainedOptimizer
constrained_optimizer = cooper.ExtrapolationConstrainedOptimizer(formulation, primal_optimizer, dual_optimizer)
# Here is the actual training loop.
# The steps follow closely the `loss -> backward -> step` Pytorch workflow.
for iter_num in range(5000):
    constrained_optimizer.zero_grad()
    lagrangian = formulation.composite_objective(cmp.closure, probs)
    formulation.custom_backward(lagrangian)
    constrained_optimizer.step(cmp.closure, probs)
```

## Installation

### Basic Installation

```bash
pip install git+https://github.com/cooper-org/cooper.git
```

### Development Installation

First, clone the [repository](https://github.com/cooper-org/cooper), navigate
to the **Cooper** root directory and install the package in development mode by running:

| Setting     | Command                                  | Notes                                     |
| ----------- | ---------------------------------------- | ----------------------------------------- |
| Development | `pip install --editable ".[dev, tests]"` | Editable mode. Matches test environment.  |
| Docs        | `pip install --editable ".[docs]"`       | Used to re-generate the documentation.    |
| Tutorials   | `pip install --editable ".[examples]"`   | Install dependencies for running examples |
| No Tests    | `pip install --editable .`               | Editable mode, without tests.             |

## Package structure

-   `cooper` - base package
    -   `problem` - abstract class for representing ConstrainedMinimizationProblems (CMPs)
    -   `constrained_optimizer` - `torch.optim.Optimizer`-like class for handling CMPs
    -   `lagrangian_formulation` - Lagrangian formulation of a CMP
    -   `multipliers` - utility class for Lagrange multipliers
    -   `optim` - aliases for Pytorch optimizers and [extra-gradient versions](https://github.com/GauthierGidel/Variational-Inequality-GAN/blob/master/optim/extragradient.py) of SGD and Adam
-   `tests` - unit tests for `cooper` components
-   `tutorials` - source code for examples contained in the tutorial gallery

## Contributions

Please read our [CONTRIBUTING](https://github.com/cooper-org/cooper/tree/master/.github/CONTRIBUTING.md)
guide prior to submitting a pull request. We use `black` for formatting, `isort`
for import sorting, `flake8` for linting, and `mypy` for type checking.

We test all pull requests. We rely on this for reviews, so please make sure any
new code is tested. Tests for `cooper` go in the `tests` folder in the root of
the repository.

## License

**Cooper** is distributed under an MIT license, as found in the
[LICENSE](https://github.com/cooper-org/cooper/tree/master/LICENSE) file.

## Acknowledgements

**Cooper** supports the use of extra-gradient style optimizers for solving the
min-max Lagrangian problem. We include the implementations of the
[extra-gradient version](https://github.com/GauthierGidel/Variational-Inequality-GAN/blob/master/optim/extragradient.py)
of SGD and Adam by Hugo Berard.

We thank Manuel del Verme for insightful discussions during the early stages of
this library.

This README follows closely the style of the [NeuralCompression](https://github.com/facebookresearch/NeuralCompression)
repository.

## How to cite this work?

If you find **Cooper** useful in your research, please consider citing it using
the snippet below:

```bibtex
@misc{gallegoPosada2022cooper,
    author={Gallego-Posada, Jose and Ramirez, Juan},
    title={Cooper: a toolkit for Lagrangian-based constrained optimization},
    howpublished={\url{https://github.com/cooper-org/cooper}},
    year={2022}
}
```
