# Cooper

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/gallego-posada/cooper/tree/master/LICENSE)
[![DOCS](https://readthedocs.org/projects/torch-cooper/badge/?version=latest)](https://torch-cooper.readthedocs.io/en/latest/?version=latest)

## About

**Cooper** is a toolkit for Lagrangian-based constrained optimization in Pytorch.
This library aims to encourage and facilitate the study of constrained
optimization problems in machine learning by providing a seamless integration
with Pytorch, while preserving the `loss -> backward -> step` workflow commonly used in many machine/deep learning pipelines.

**Cooper** is under active development and future API changes might break backward compatibility.

## Getting Started

Here we consider a simple convex optimization problem to illustrate how to use **Cooper**.
This example is inspired by [this StackExchange question](https://datascience.stackexchange.com/questions/107366/how-do-you-solve-strictly-constrained-optimization-problems-with-pytorch):
>*I am trying to solve the following problem using Pytorch: given a 6-sided die whose
>average roll is known to be 4.5, what is the maximum entropy distribution for the faces?*

```python
import torch
import cooper

class MaximumEntropy(cooper.ConstrainedMinimizationProblem):
    def __init__(self, mean_constraint):
        self.mean_constraint = mean_constraint
        super().__init__(is_constrained=True)

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
dual_optimizer = cooper.optim.partial(cooper.optim.ExtraSGD, lr=9e-3, momentum=0.7)

# Wrap the formulation and both optimizers inside a ConstrainedOptimizer
coop = cooper.ConstrainedOptimizer(formulation, primal_optimizer, dual_optimizer)

# Here is the actual training loop
for iter_num in range(5000):
    coop.zero_grad()
    lagrangian = formulation.composite_objective(cmp.closure, probs)
    formulation.custom_backward(lagrangian)
    coop.step(cmp.closure, probs)
```

## Installation

**Cooper** is a project currently under development. You can install the
repository in development mode.

### Basic Installation

```bash
pip install git@github.com:gallego-posada/cooper.git#egg=cooper
```

### Development Installation

First, clone the repository and navigate to the **Cooper** root
directory and install the package in development mode by running:

```bash
pip install --editable ".[dev]"
```

If you are not interested in matching the test environment, you can just
apply:
```bash
pip install --editable .
```

## Cooper

- `cooper` - base package
    - `problem` - abstract class for representing CMPs
    - `constrained_optimizer` - Pytorch optimizer class for handling constrained minimization problems (CMPs)
    - `lagrangian_formulation` - Lagrangian formulation of a CMP
    - `multipliers` - utility class for Lagrange multipliers
    - `optim` - aliases for Pytorch optimizers and [extra-gradient versions](https://github.com/GauthierGidel/Variational-Inequality-GAN/blob/master/optim/extragradient.py) of SGD and Adam

## Tutorial Notebooks

The `tutorials` directory contains interactive notebooks which showcase the core
features of the toolkit. Existing tutorials are:

- Tutorials TBD

## Contributions

Please read our [CONTRIBUTING](https://github.com/gallego-posada/cooper/tree/master/.github/CONTRIBUTING.md) guide prior to submitting a pull request.

We test all pull requests. We rely on this for reviews, so please make sure any
new code is tested. Tests for `cooper` go in the `tests` folder in
the root of the repository.

We use `black` for formatting, `isort` for import sorting, `flake8` for
linting, and `mypy` for type checking.

## License

**Cooper** is distributed under an MIT license, as found in the [LICENSE](https://github.com/gallego-posada/cooper/tree/master/LICENSE) file.

## Acknowledgements

We thank Manuel del Verme for insightful discussions in the early stages of this
library.

Many of the structural design ideas behind **Cooper** are heavily inspired by the
[TensorFlow Constrained Optimization (TFCO)](https://github.com/google-research/tensorflow_constrained_optimization)
library. We highly recommend TFCO for TensorFlow-based projects and will continue
to integrate more of TFCO's feature in future releases.

**Cooper** supports the use of extra-gradient style optimizers for solving the min-max
Lagrangian problem. We include the implementations of the
[extra-gradient version](https://github.com/GauthierGidel/Variational-Inequality-GAN/blob/master/optim/extragradient.py)
of SGD and Adam by Hugo Berard.

This README follows closely the style of the [NeuralCompression](https://github.com/facebookresearch/NeuralCompression)
repository.

## How to cite this work?

If you find **Cooper** useful in your work, please consider citing it using the snippet below:

```bibtex
@misc{gallegoPosada2022cooper,
    author={Gallego-Posada, Jose and Ramirez, Juan},
    title={Cooper: a toolkit for Lagrangian-based constrained optimization},
    howpublished={\url{https://github.com/gallego-posada/cooper}},
    year={2022}
}
```
