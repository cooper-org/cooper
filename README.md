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

Here we consider a simple convex constrained optimization problem that involves
training a Logistic Regression clasifier on the MNIST dataset. The model is
constrained so that the squared L2 norm of its parameters is less than 1.

This example illustrates how **Cooper** integrates with:
- constructing a ``cooper.LagrangianFormulation`` and a ``cooper.SimultaneousOptimizer``
- models defined using a ``torch.nn.Module``,
- CUDA acceleration,
- typical machine learning training loops,
- extracting the value of the Lagrange multipliers from a ``cooper.LagrangianFormulation``.

Please visit the entry in the **Tutorial Gallery** for a complete version of the code.

```python
import cooper
import torch

train_loader = ... # Create a Pytorch Dataloader for MNIST
loss_fn = torch.nn.CrossEntropyLoss()

# Create a Logistic Regression model
model = torch.nn.Linear(in_features=28 * 28, out_features=10, bias=True)
if torch.cuda.is_available():
    model = model.cuda()
primal_optimizer = torch.optim.Adagrad(model.parameters(), lr=5e-3)

# Create a Cooper formulation, and pick a Pytorch optimizer class for the dual variables
formulation = cooper.LagrangianFormulation()
dual_optimizer = cooper.optim.partial_optimizer(torch.optim.SGD, lr=1e-3)

# Create a ConstrainedOptimizer for performing simultaneous updates based on the
# formulation, and the selected primal and dual optimizers.
cooper_optimizer = cooper.SimultaneousOptimizer(
    formulation, primal_optimizer, dual_optimizer
)

for epoch_num in range(50):
    for batch_num, (inputs, targets) in enumerate(train_loader):

        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

        logits = model.forward(inputs.view(inputs.shape[0], -1))
        loss = loss_fn(logits, targets)

        sq_l2_norm = model.weight.pow(2).sum() + model.bias.pow(2).sum()
        # Constraint defects use convention “g - \epsilon ≤ 0”
        constraint_defect = sq_l2_norm - 1.0

        # Create a CMPState object, which contains the loss and constraint defect
        cmp_state = cooper.CMPState(loss=loss, ineq_defect=constraint_defect)

        cooper_optimizer.zero_grad()
        lagrangian = formulation.compute_lagrangian(pre_computed_state=cmp_state)
        formulation.backward(lagrangian)
        cooper_optimizer.step()

    # We can extract the value of the Lagrange multiplier for the constraint
    # The dual variables are stored and updated internally by Cooper
    lag_multiplier, _ = formulation.state()

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

## Projects built with Cooper

- J. Gallego-Posada et al. Controlled Sparsity via Constrained Optimization or: How I Learned to Stop Tuning Penalties and Love Constraints. In [NeurIPS 2022](https://arxiv.org/abs/2208.04425).
- S. Lachapelle and S. Lacoste-Julien. Partial Disentanglement via Mechanism Sparsity. In [CLR Workshop at UAI 2022](https://arxiv.org/abs/2207.07732).
- J. Ramirez and J. Gallego-Posada. L0onie: Compressing COINS with L0-constraints. In [Sparsity in Neural Networks Workshop 2022](https://arxiv.org/abs/2207.04144).

*If you would like your work to be highlighted in this list, please open a pull request.*

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
