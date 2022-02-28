# Cooper

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/gallego-posada/cooper/tree/master/LICENSE)

[![DOCS](https://readthedocs.org/projects/torch-cooper/badge/?version=latest)](https://torch-cooper.readthedocs.io/en/latest/?version=latest)

## About

Cooper is a toolkit for Lagrangian-based constrained optimization in Pytorch.
This library aims to encourage and facilitate the study of constrained
optimization problems in machine learning by providing a seamless integration
with Pytorch, while preserving the `loss -> backward -> step` workflow commonly used in many machine/deep learning pipelines.

Cooper is under active development and future API changes might break backward compatibility.

## Getting Started


```python
import torch
import cooper

TBC

```
## Installation

Cooper is a project currently under development. You can install the
repository in development mode.

### PyPI Installation

First, install PyTorch according to the directions from the
[PyTorch website](https://pytorch.org/). Then, you should be able to run

```bash
pip install git@github.com:gallego-posada/cooper.git#egg=cooper
```

### Development Installation

First, clone the repository and navigate to the Cooper root
directory and install the package in development mode by running:

```bash
pip install --editable ".[dev]"
```

If you are not interested in matching the test environment, then you can just
apply `pip install -e .`.



## Cooper

- `cooper` - base package
    - `constrained_optimizer` - Pytorch optimizer class for handling constrained minimization problems (CMPs)
    - `lagrangian_formulation` - Lagrangian formulation of a CMP
    - `multipliers` - utility class for Lagrange multipliers
    - `optim` - aliases for Pytorch optimizers and [extra-gradient versions](https://github.com/GauthierGidel/Variational-Inequality-GAN/blob/master/optim/extragradient.py) of SGD and Adam
    - `problem` - abstract class for representing CMPs

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

Cooper is distributed under an MIT license, as found in the [LICENSE](https://github.com/gallego-posada/cooper/tree/master/LICENSE) file.

## Acknowledgements

We thank Manuel del Verme for insightful discussions in the early stages of this
library.

Many of the structural design ideas behind cooper are heavily inspired by the
brilliant [TensorFlow Constrained Optimization (TFCO)](https://github.com/google-research/tensorflow_constrained_optimization)
library. We highly recommend TFCO for TensorFlow-based projects and will continue
to integrate more of TFCO's feature in future releases.

Cooper supports the use of extra-gradient style optimizers for solving the min-max Lagrangian problem. We include the implementations of the [extra-gradient version](https://github.com/GauthierGidel/Variational-Inequality-GAN/blob/master/optim/extragradient.py) of SGD and Adam by Hugo Berard.

This README follows closely the style of the [NeuralCompression](https://github.com/facebookresearch/NeuralCompression) repository.
## Cite

If you find cooper useful in your work, please use the citation snippet below:

```bibtex
@misc{gallegoPosada2022cooper,
    author={Gallego-Posada, Jose and Ramirez, Juan},
    title={Cooper: a toolkit for Lagrangian-based constrained optimization},
    howpublished={\url{https://github.com/gallego-posada/cooper}},
    year={2022}
}
```
