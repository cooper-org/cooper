# Cooper

[![LICENSE](https://img.shields.io/pypi/l/cooper-optim)](https://github.com/cooper-org/cooper/tree/master/LICENSE)
[![Version](https://img.shields.io/pypi/v/cooper-optim?label=version)](https://pypi.python.org/pypi/cooper-optim)
[![Downloads](https://img.shields.io/pepy/dt/cooper-optim?color=blue)](https://pypi.python.org/pypi/cooper-optim)
[![Python](https://img.shields.io/pypi/pyversions/cooper-optim?label=Python&logo=python&logoColor=white)](https://pypi.python.org/pypi/cooper-optim)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1+-EE4C2C?logo=pytorch)](https://pytorch.org/docs/stable/index.html)
[![DOCS](https://img.shields.io/readthedocs/cooper)](https://cooper.readthedocs.io/en/latest/?version=latest)
[![Coverage badge](https://raw.githubusercontent.com/cooper-org/cooper/python-coverage-comment-action-data/badge.svg)](https://github.com/cooper-org/cooper/tree/python-coverage-comment-action-data)
[![Continuous Integration](https://github.com/cooper-org/cooper/actions/workflows/ci.yml/badge.svg)](https://github.com/cooper-org/cooper/actions/workflows/ci.yml)
[![Stars](https://img.shields.io/github/stars/cooper-org/cooper)](https://github.com/cooper-org/cooper)
[![HitCount](https://img.shields.io/endpoint?url=https://hits.dwyl.com/cooper-org/cooper.json&color=brightgreen)](https://cooper.readthedocs.io/en/latest/?version=latest)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)](https://github.com/cooper-org/cooper/issues)
[![Discord](https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/Aq5PjH8m6E)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## What is Cooper?

**Cooper** is an open-source library for solving constrained
optimization problems in PyTorch.

**Cooper** implements several Lagrangian-based first-order update schemes that are applicable to a wide range of continuous constrained optimization problems. Although **Cooper** is specifically designed for deep learning applications where gradients are estimated based on mini-batches, it is suitable for general continuous constrained optimization.

**Cooper** was born out of the need to handle constrained optimization problems for which the objective or constraints are not necessarily "nicely behaved" or "theoretically tractable," e.g., when no (efficient) projection or proximal term are available. These assumptions have helped create libraries like [CHOP](https://github.com/openopt/chop) and [GeoTorch](https://github.com/Lezcano/geotorch), but they are often not met in modern machine learning problems.

TODO: mention Cooper MLOSS paper

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

### Quick Start

To use **Cooper**, you need to:

- Implement a `cooper.ConstrainedMinimizationProblem` (CMP) class. It should hold `cooper.Constraint` objects, each associated with a corresponding `cooper.Multiplier`. It should also implement a `compute_cmp_state` method that computes the objective value and constraint violations, packaged in a `cooper.CMPState` object.
- Create a `torch.optim.Optimizer` for the primal variables and a `torch.optim.Optimizer` for the dual variables (with `maximize=True`).
- Wrap the primal and dual optimizers in a `cooper.optim.CooperOptimizer` (such as `cooper.optim.SimultaneousOptimizer` for simultaneous updates).
- Use the `roll` method of the `cooper.optim.CooperOptimizer` to perform updates. It internally calls the `compute_cmp_state` method of the CMP and computes the Lagrangian (forward pass), calls `backward`, performs the primal and dual optimizer steps, and calls `zero_grad` on both optimizers.

### Example

This is an abstract example on how to solve a constrained optimization problem with
**Cooper**. You can find runnable notebooks in our [**Tutorials**](https://cooper.readthedocs.io/en/master/notebooks/index.html).

```python
import cooper
import torch


class MyCMP(cooper.ConstrainedMinimizationProblem):
    def __init__(self):
        super().__init__()
        multiplier = cooper.multipliers.DenseMultiplier(num_constraints=..., device=...)
        # By default, constraints are built using `formulation_type=cooper.LagrangianFormulation`
        self.constraint = cooper.Constraint(
            multiplier=multiplier, constraint_type=cooper.ConstraintType.INEQUALITY
        )

    def compute_cmp_state(self, model, inputs, targets):
        loss = ...
        constraint_state = cooper.ConstraintState(violation=...)
        observed_constraints = {self.constraint: constraint_state}

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
        # `roll` is a convenience function that packages together the evaluation
        # of the loss, call for gradient computation, the primal and dual updates and zero_grad
        compute_cmp_state_kwargs = {"model": model, "inputs": inputs, "targets": targets}
        roll_out = cooper_optimizer.roll(compute_cmp_state_kwargs=compute_cmp_state_kwargs)
        # `roll_out` is a namedtuple containing the loss, last CMPState, and the primal
        # and dual Lagrangian stores, useful for inspection and logging
```

## Contributions

Please read our [CONTRIBUTING](https://github.com/cooper-org/cooper/tree/master/.github/CONTRIBUTING.md) guide prior to submitting a pull request. We use `ruff` for formatting and linting, and `mypy` for type checking.

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

**Cooper**'s FAQ is available [here](https://cooper.readthedocs.io/en/latest/faq.html).
