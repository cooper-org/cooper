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

TODO:

**Cooper** is a toolkit for Lagrangian-based constrained optimization in PyTorch.
This library aims to encourage and facilitate the study of constrained
optimization problems in machine learning.

**Cooper** is (almost!) seamlessly integrated with PyTorch and preserves the
usual `loss -> backward -> step` workflow. If you are already familiar with
PyTorch, using **Cooper** will be a breeze! 🙂

**Cooper** was born out of the need to handle constrained optimization problems
for which the loss or constraints are not necessarily "nicely behaved"
or "theoretically tractable," e.g., when no (efficient) projection or proximal
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
        # `roll_out` is a struct containing the loss, last CMPState, and the primal
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

TODO: subsections here?
TODO: Have a separate FAQ page?
TODO: emojis?

<details>
  <summary style="font-size: 1.2rem;">
    What types of problems can I solve with <b>Cooper</b>?
  </summary>
  <div style="margin-left: 20px;">
    Answer here. For convex problems or problems with special structure, suggest other libraries.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    Where can I get help with <b>Cooper</b>?
  </summary>
  <div style="margin-left: 20px;">
    You can ask questions and get help on our <a href="https://discord.gg/Aq5PjH8m6E">Discord server</a>.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    Where can I learn more about constrained optimization?
  </summary>
  <div style="margin-left: 20px;">
    You can find more on convex constrained optimization in the book <a href="https://web.stanford.edu/~boyd/cvxbook/">Convex Optimization</a> by Boyd and Vandenberghe.
    For non-convex constrained optimization, you can check out the book <a href="http://athenasc.com/nonlinbook.html">Nonlinear Programming</a> by Bertsekas.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    What kind of problems can I solve with <b>Cooper</b>?
  </summary>
  <div style="margin-left: 20px;">
    <b>Cooper</b> is designed to solve constrained optimization problems in machine learning.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    What problem formulations does <b>Cooper</b> support?
  </summary>
  <div style="margin-left: 20px;">
    <b>Cooper</b> supports the following formulations:
    <ul>
      <li>Lagrangian Formulation</li>
      <li>Augmented Lagrangian Formulation</li>
    </ul>
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    When should I pick any of these formulations?
  </summary>
  <div style="margin-left: 20px;">
    <b>Lagrangian Formulation</b> is a good choice when ...
    <br>
    <b>Augmented Lagrangian Formulation</b> is a good choice when ...
  </div>
</details>

<details>
  <summary>
    What is a good starting configuration for a Cooper optimizer (primal and dual)?
  </summary>
    For the dual optimizer, we recommend using SGD with a learning rate not too high to avoid overshoots and setting `maximize=True`.
    <br>
    For the primal optimizer, we recommend ...
</details>

<details>
  <summary style="font-size: 1.2rem;">
    Which <b>Cooper</b> optimizer should I use?
  </summary>
  <div style="margin-left: 20px;">
    <b>Cooper</b> provides a range of optimizers to choose from. The <b>AlternatingDualPrimalOptimizer</b> is a good starting point.
  </div>
</details>

### Debugging and troubleshooting

<details>
  <summary style="font-size: 1.2rem;">
    Why is my problem not becoming feasible?
  </summary>
  <div style="margin-left: 20px;">
    Answer here.
  </div>
</details>


<details>
  <summary style="font-size: 1.2rem;">
    Why is my objective function increasing? 😟
  </summary>
  <div style="margin-left: 20px;">
    There are several reasons why this might happen. But the most common one is that the dual learning rate is too high. Try reducing it.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    How can I tell if <b>Cooper</b> found a "good" solution?
  </summary>
  <div style="margin-left: 20px;">
    Answer here.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    What quantities should I log for sanity-checking?
  </summary>
  <div style="margin-left: 20px;">
    Answer here.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    What do typical multiplier dynamics look like?
  </summary>
  <div style="margin-left: 20px;">
    Answer here.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    What should I do if my Lagrange multipliers diverge?
  </summary>
  <div style="margin-left: 20px;">
    Answer here.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    What should I do if my Lagrange multipliers oscillate too much?
  </summary>
  <div style="margin-left: 20px;">
    Answer here.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    What should I do if my Lagrange multipliers are too noisy?
  </summary>
  <div style="margin-left: 20px;">
    Answer here.
  </div>
</details>

### Computational considerations

<details>
  <summary style="font-size: 1.2rem;">
    Does <b>Cooper</b> support GPU acceleration?
  </summary>
  <div style="margin-left: 20px;">
    Answer here.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    Does <b>Cooper</b> support DDP execution?
  </summary>
  <div style="margin-left: 20px;">
    Answer here.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    Does <b>Cooper</b> support AMP?
  </summary>
  <div style="margin-left: 20px;">
    Answer here.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    What if my problem has a lot of constraints?
  </summary>
  <div style="margin-left: 20px;">
    Answer here. IndexedMultipliers, ImplicitMultipliers, etc.
  </div>

### Advanced topics


### Miscellaneous

<details>
  <summary style="font-size: 1.2rem;">
    How do I cite <b>Cooper</b>?
  </summary>
  <div style="margin-left: 20px;">
    Answer here.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    Is there a JAX version of <b>Cooper</b>?
  </summary>
  <div style="margin-left: 20px;">
    Answer here.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    Is there a TensorFlow version of <b>Cooper</b>?
  </summary>
  <div style="margin-left: 20px;">
    Answer here. TFCO is a good alternative.
  </div>
</details>

If non convex
Or stochastic
Autograd differentiable objective and constraints (or non-differentiable constraints but with a surrogate)
Something about CMPState data structure
Argue for cheap cost (for free, compared to general minmax game)
Gradient of primal Lagrangian is autograd-friendly
Gradient of a linear combination of functions
Why are they useful?
What should I do if they oscillate too much?
What if they don’t stabilize/converge?
Complementary slackness
Dynamics/Solution
Loss/Lagrangian/ConstraintViolation
