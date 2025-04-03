# **Cooper**

[![LICENSE](https://img.shields.io/pypi/l/cooper-optim)](https://github.com/cooper-org/cooper/tree/main/LICENSE)
[![Version](https://img.shields.io/pypi/v/cooper-optim?label=version)](https://pypi.python.org/pypi/cooper-optim)
[![Downloads](https://img.shields.io/pepy/dt/cooper-optim?color=blue)](https://pypi.python.org/pypi/cooper-optim)
[![Python](https://img.shields.io/pypi/pyversions/cooper-optim?label=Python&logo=python&logoColor=white)](https://pypi.python.org/pypi/cooper-optim)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1+-EE4C2C?logo=pytorch)](https://pytorch.org/docs/stable/index.html)
[![DOCS](https://img.shields.io/readthedocs/cooper)](https://cooper.readthedocs.io/en/latest/?version=latest)
[![Coverage badge](https://raw.githubusercontent.com/cooper-org/cooper/python-coverage-comment-action-data/badge.svg)](https://github.com/cooper-org/cooper/tree/python-coverage-comment-action-data)
[![Continuous Integration](https://github.com/cooper-org/cooper/actions/workflows/ci.yaml/badge.svg)](https://github.com/cooper-org/cooper/actions/workflows/ci.yaml)
[![Stars](https://img.shields.io/github/stars/cooper-org/cooper)](https://github.com/cooper-org/cooper)
[![HitCount](https://hits.sh/github.com/cooper-org/cooper.svg)](https://cooper.readthedocs.io/en/latest/?version=latest)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)](https://github.com/cooper-org/cooper/issues)
[![Discord](https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/Aq5PjH8m6E)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## What is **Cooper**?

**Cooper** is a library for solving constrained optimization problems in [PyTorch](https://github.com/pytorch/pytorch).

**Cooper** implements several Lagrangian-based (first-order) update schemes that are applicable to a wide range of continuous constrained optimization problems. **Cooper** is mainly targeted for deep learning applications, where gradients are estimated based on mini-batches, but it is also suitable for general continuous constrained optimization tasks.

There exist other libraries for constrained optimization in PyTorch, like [CHOP](https://github.com/openopt/chop) and [GeoTorch](https://github.com/Lezcano/geotorch), but they rely on assumptions about the constraints (such as admitting efficient projection or proximal operators). These assumptions are often not met in modern machine learning problems. **Cooper** can be applied to a wider range of constrained optimization problems (including non-convex problems) thanks to its Lagrangian-based approach.

You can check out **Cooper**'s FAQ [here](#faq).

**Cooper**'s companion paper is available [here](https://arxiv.org/abs/2504.01212).

- [**Cooper**](#cooper)
  - [What is **Cooper**?](#what-is-cooper)
  - [Installation](#installation)
  - [Getting Started](#getting-started)
    - [Quick Start](#quick-start)
    - [Example](#example)
  - [Contributions](#contributions)
  - [Acknowledgements](#acknowledgements)
  - [License](#license)
  - [How to cite **Cooper**](#how-to-cite-cooper)


## Installation

To install the latest release of **Cooper**, use the following command:

```bash
pip install cooper-optim
```

To install the latest **development** version, use the following command instead:

```bash
pip install git+https://github.com/cooper-org/cooper@main
```

## Getting Started


### Quick Start

To use **Cooper**, you need to:

- Implement a [`ConstrainedMinimizationProblem`](https://cooper.readthedocs.io/en/latest/problem.html#cooper.ConstrainedMinimizationProblem) (CMP) class and its associated [`ConstrainedMinimizationProblem.compute_cmp_state`](https://cooper.readthedocs.io/en/latest/problem.html#cooper.ConstrainedMinimizationProblem.compute_cmp_state) method. This method computes the value of the objective function and constraint violations, and packages them in a [`CMPState`](https://cooper.readthedocs.io/en/latest/problem.html#cooper.CMPState) object.
- The initialization of the [`CMP`](https://cooper.readthedocs.io/en/latest/problem.html#cooper.ConstrainedMinimizationProblem) must create a [`Constraint`](https://cooper.readthedocs.io/en/latest/problem.html#cooper.constraints.Constraint) object for each constraint. It is necessary to specify a formulation type (e.g. [`Lagrangian`](https://cooper.readthedocs.io/en/latest/formulations.html#cooper.formulations.Lagrangian)). Finally, if the chosen formulation requires it, each constraint needs an associated [`Multiplier`](https://cooper.readthedocs.io/en/latest/multipliers.html) object corresponding to the Lagrange multiplier for that constraint.
- Create a [`torch.optim.Optimizer`](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer) for the primal variables and a [`torch.optim.Optimizer(maximize=True)`](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer) for the dual variables (i.e. the multipliers). Then, wrap these two optimizers in a [`cooper.optim.CooperOptimizer`](https://cooper.readthedocs.io/en/latest/optim.html#cooper.optim.CooperOptimizer) (such as [`SimultaneousOptimizer`](https://cooper.readthedocs.io/en/latest/optim.html#cooper.optim.SimultaneousOptimizer) for executing simultaneous primal-dual updates).
- You are now ready to perform updates on the primal and dual parameters using the [`CooperOptimizer.roll()`](https://cooper.readthedocs.io/en/latest/optim.html#cooper.optim.CooperOptimizer.roll) method. This method triggers the following calls:
  - `zero_grad()` on both optimizers,
  - [`compute_cmp_state()`](https://cooper.readthedocs.io/en/latest/problem.html#cooper.ConstrainedMinimizationProblem.compute_cmp_state) on the [`CMP`](https://cooper.readthedocs.io/en/latest/problem.html#cooper.ConstrainedMinimizationProblem),
  - compute the Lagrangian based on the latest [`CMPState`](https://cooper.readthedocs.io/en/latest/problem.html#cooper.CMPState),
  - `backward()` on the Lagrangian,
  - [`step()`](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html#torch.optim.Optimizer.step) on both optimizers.
- To access the value of the loss, constraint violations, and Lagrangian terms, you can inspect the returned [`RollOut`](https://cooper.readthedocs.io/en/latest/optim.html#cooper.optim.RollOut) object from the call to [`roll()`](https://cooper.readthedocs.io/en/latest/optim.html#cooper.optim.CooperOptimizer.roll).

### Example

This is an abstract example on how to solve a constrained optimization problem with
**Cooper**. You can find runnable notebooks with concrete examples in our [**Tutorials**](https://cooper.readthedocs.io/en/latest/notebooks/index.html).

```python
import cooper
import torch

# Set up GPU acceleration
DEVICE = ...

class MyCMP(cooper.ConstrainedMinimizationProblem):
    def __init__(self):
        super().__init__()
        multiplier = cooper.multipliers.DenseMultiplier(num_constraints=..., device=DEVICE)
        # By default, constraints are built using `formulation_type=cooper.formulations.Lagrangian`
        self.constraint = cooper.Constraint(
            multiplier=multiplier, constraint_type=cooper.ConstraintType.INEQUALITY
        )

    def compute_cmp_state(self, model, inputs, targets):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        loss = ...
        constraint_state = cooper.ConstraintState(violation=...)
        observed_constraints = {self.constraint: constraint_state}

        return cooper.CMPState(loss=loss, observed_constraints=observed_constraints)


train_loader = ...
model = (...).to(DEVICE)
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

We appreciate all contributions. Please let us know if you encounter a bug by [filing an issue](https://github.com/cooper-org/cooper/issues).

If you plan to contribute new features, utility functions, or extensions, please first open an issue and discuss the feature with us. To learn more about making a contribution to **Cooper**, please see our [Contribution page](https://cooper.readthedocs.io/en/latest/CONTRIBUTING.html).

## Papers Using **Cooper**

**Cooper** has enabled several papers published at top machine learning conferences: [Gallego-Posada et al. (2022)](https://arxiv.org/abs/2208.04425); [Lachapelle and Lacoste-Julien (2022)](https://arxiv.org/abs/2207.07732); [Ramirez and Gallego-Posada (2022)](https://arxiv.org/abs/2207.04144); [Zhu et al. (2023)](https://arxiv.org/abs/2310.08106); [Hashemizadeh et al. (2024)](https://arxiv.org/abs/2310.20673); [Sohrabi et al. (2024)](https://arxiv.org/abs/2406.04558); [Lachapelle et al. (2024)](https://arxiv.org/abs/2401.04890); [Jang et al. (2024)](https://arxiv.org/abs/2312.10289); [Navarin et al. (2024)](https://ieeexplore.ieee.org/document/10650578); [Chung et al. (2024)](https://arxiv.org/abs/2404.01216).


## Acknowledgements

We thank Manuel Del Verme, Daniel Otero, and Isabel Urrego for useful discussions during the early stages of **Cooper**.

Many **Cooper** features arose during the development of several research papers. We would like to thank our co-authors Yoshua Bengio, Juan Elenter, Akram Erraqabi, Golnoosh Farnadi, Ignacio Hounie, Alejandro Ribeiro, Rohan Sukumaran, Motahareh Sohrabi and Tianyue (Helen) Zhang.

## License

**Cooper** is distributed under an MIT license, as found in the
[LICENSE](https://github.com/cooper-org/cooper/tree/main/LICENSE) file.

## How to cite **Cooper**

To cite **Cooper**, please cite [this paper](https://arxiv.org/abs/2504.01212):

```bibtex
@article{gallegoPosada2025cooper,
    author={Gallego-Posada, Jose and Ramirez, Juan and Hashemizadeh, Meraj and Lacoste-Julien, Simon},
    title={{Cooper: A Library for Constrained Optimization in Deep Learning}},
    journal={arXiv preprint arXiv:2504.01212},
    year={2025}
}
```
