# Cooper

[![LICENSE](https://img.shields.io/pypi/l/cooper-optim)](https://github.com/cooper-org/cooper/tree/master/LICENSE)
[![Version](https://img.shields.io/pypi/v/cooper-optim?label=version)](https://pypi.python.org/pypi/cooper-optim)
[![Downloads](https://static.pepy.tech/badge/cooper-optim)](https://pypi.python.org/pypi/cooper-optim)
[![Python](https://img.shields.io/pypi/pyversions/cooper-optim.svg?style=flat&logo=python&logoColor=white&label=Python)](https://pypi.python.org/pypi/cooper-optim)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/docs/stable/index.html)
[![DOCS](https://readthedocs.org/projects/cooper/badge/?version=latest)](https://cooper.readthedocs.io/en/latest/?version=latest)
[![Coverage badge](https://raw.githubusercontent.com/cooper-org/cooper/python-coverage-comment-action-data/badge.svg)](https://github.com/cooper-org/cooper/tree/python-coverage-comment-action-data)
[![Continuous Integration](https://github.com/cooper-org/cooper/actions/workflows/ci.yml/badge.svg)](https://github.com/cooper-org/cooper/actions/workflows/ci.yml)
[![Stars](https://img.shields.io/github/stars/cooper-org/cooper)](https://github.com/cooper-org/cooper)
[![HitCount](https://img.shields.io/endpoint?url=https://hits.dwyl.com/cooper-org/cooper.json&color=brightgreen)](https://cooper.readthedocs.io/en/latest/?version=latest)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/cooper-org/cooper/issues)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/Aq5PjH8m6E)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## What is Cooper?

TODO:

**Cooper** is a toolkit for Lagrangian-based constrained optimization in PyTorch.
This library aims to encourage and facilitate the study of constrained
optimization problems in machine learning.

**Cooper** is (almost!) seamlessly integrated with PyTorch and preserves the
usual `loss -> backward -> step` workflow. If you are already familiar with
PyTorch, using **Cooper** will be a breeze! 🙂

**Cooper** was born out of the need to handle constrained optimization problems
for which the loss or constraints are not necessarily "nicely behaved"
or "theoretically tractable", e.g. when no (efficient) projection or proximal
are available. Although assumptions of this kind have enabled the development of
great PyTorch-based libraries such as [CHOP](https://github.com/openopt/chop)
and [GeoTorch](https://github.com/Lezcano/geotorch), they are seldom satisfied
in the context of many modern machine learning problems.

Many of the structural design ideas behind **Cooper** are heavily inspired by
the [TensorFlow Constrained Optimization (TFCO)](https://github.com/google-research/tensorflow_constrained_optimization)
library. We highly recommend TFCO for TensorFlow-based projects and will
continue to integrate more of TFCO's features in future releases.

⚠️ This library is under active development. Future API changes might break backward
compatibility. ⚠️

TODO: mention MLOSS paper
TODO: mention Cooper poster?

| [**Installation**](#installation) | [**Getting Started**](#getting-started) | [**Package Structure**](#package-structure) | [**Contributions**](#contributions) | [**Acknowledgements**](#acknowledgements) | [**License**](#license) | [**How to cite Cooper**](#how-to-cite-cooper) | [**FAQ**](#faq) |


## Installation

To install the latest release of Cooper, use the following command:

```bash
pip install cooper-optim
```

To install the latest **development** version, use the following command instead:

```bash
pip install git+https://github.com/cooper-org/cooper
```

## Getting Started

This is an abstract example on how to solve a constrained optimization problem with
**Cooper**. You can find runnable notebooks in our [**Tutorials**](#TODO).

[comment]: <The user implements a \texttt{ConstrainedMinimization-} \texttt{Problem} (\CMP) holding \texttt{Constraint} objects, each in turn holding a corresponding \texttt{Multiplier}. The \CMP's \texttt{compute\_cmp\_state} method returns the objective value and constraints violations, stored in a \texttt{CMPState} dataclass. \texttt{CooperOptimizer}s wrap the primal and dual optimizers and perform updates (such as simultaneous GDA). The \texttt{roll} method of \texttt{CooperOptimizer}s is a convenience function to (i) perform a \texttt{zero\_grad} on all optimizers, (ii) compute the Lagrangian, (iii) call its \texttt{backward} and (iv) perform the primal and dual optimizer steps.>

-   `cooper` - base package
    -   `problem` - abstract class for representing ConstrainedMinimizationProblems (CMPs)
    -   `constrained_optimizer` - `torch.optim.Optimizer`-like class for handling CMPs
    -   `lagrangian_formulation` - Lagrangian formulation of a CMP
    -   `multipliers` - utility class for Lagrange multipliers
    -   `optim` - aliases for PyTorch optimizers and [extra-gradient versions](https://github.com/GauthierGidel/Variational-Inequality-GAN/blob/master/optim/extragradient.py) of SGD and Adam
-   `tests` - unit tests for `cooper` components
-   `tutorials` - source code for examples contained in the tutorial gallery


```python
import cooper
import torch


class MyCMP(cooper.ConstrainedMinimizationProblem)
    def __init__(self):
        multiplier = cooper.multipliers.DenseMultiplier(num_constraints=..., device=...)
        # By default constraints are built using `formulation_type=cooper.LagrangianFormulation`
        self.constraint = cooper.Constraint(
            multiplier=multiplier, constraint_type=cooper.ConstraintType.INEQUALITY
        )

    def compute_cmp_state(self, model, inputs, targets):
        loss = ...
        constraint_state = cooper.ConstraintState(violation=...)
        observed_constraints = {self.constraint, constraint_state}

        return cooper.CMPState(loss=loss, observed_constraints=observed_constraints)

train_loader = ...
model = ...
cmp = MyCMP()

primal_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# Must set `maximize=True` since the Lagrange multipliers solve a _maximization_ problem
dual_optimizer = torch.optim.SGD(cmp.dual_parameters(), lr=1e-2, maximize=True)

cooper_optimizer = cooper.optim.SimultaneousOptimizer(
    cmp=cmp, primal_optimizers=primal_optimizer, dual_optimizers=dual_optimizer
)

for epoch_num in range(NUM_EPOCHS):
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        # `roll` is a convenience function that packages together the evaluation
        # of the loss, call for gradient computation, the primal and dual updates and zero_grad
        compute_cmp_state_kwargs = {"model": model, "inputs": inputs, "targets": targets}
        roll_out = cooper_optimizer.roll(compute_cmp_state_kwargs=compute_cmp_state_kwargs)
        # `roll_out` is a struct containing the loss, last CMPState, and the primal
        # and dual Lagrangian stores, useful for inspection and logging
```


## Contributions

Please read our [CONTRIBUTING](https://github.com/cooper-org/cooper/tree/master/.github/CONTRIBUTING.md)
guide prior to submitting a pull request. We use `ruff` for formatting and linting, and `mypy` for type checking.

We test all pull requests. We rely on this for reviews, so please make sure any
new code is tested. Tests for `cooper` go in the `tests` folder in the root of
the repository.

### Development Installation

First, clone the [repository](https://github.com/cooper-org/cooper), navigate
to the **Cooper** root directory and install the package in development mode by running:

| Setting     | Command                                 | Notes                                           |
|-------------|-----------------------------------------|-------------------------------------------------|
| No Tests    | `pip install --editable .`              | Editable mode, without tests.                   |
| Development | `pip install --editable ".[test]"`      | Editable mode. Matches test environment.        |
| Development | `pip install --editable ".[dev]"`       | Editable mode. Matches development environment. |
| Tutorials   | `pip install --editable ".[notebooks]"` | Install dependencies for running notebooks.     |
| Docs        | `pip install --editable ".[docs]"`      | Used to generate the documentation.             |


## Acknowledgements

We thank Manuel Del Verme, Daniel Otero, and Isabel Urrego for useful discussions during the early stages of **Cooper**.

## License

**Cooper** is distributed under an MIT license, as found in the
[LICENSE](https://github.com/cooper-org/cooper/tree/master/LICENSE) file.


## How to cite **Cooper**

To cite **Cooper**, please cite [this paper](link-to-paper):

```bibtex
@misc{gallegoPosada2024cooper,
    author={Gallego-Posada, Jose and Ramirez, Juan and Hashemizadeh, Meraj and Lacoste-Julien, Simon},
    title={{Cooper: A Library for Constrained Optimization in Deep Learning}},
    howpublished={\url{https://github.com/cooper-org/cooper}},
    year={2024}
}
```


## FAQ

### ...
