"""Top-level package for Constrained Optimization in Pytorch."""

import sys

if sys.version_info >= (3, 8):
    from importlib.metadata import PackageNotFoundError, version
else:
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("cooper")
except PackageNotFoundError:
    # package is not installed
    import warnings

    warnings.warn("Could not retrieve cooper version!")

from cooper.constrained_optimizer import ConstrainedOptimizer
from cooper.formulation import (
    Formulation,
    LagrangianFormulation,
    UnconstrainedFormulation,
)
from cooper.problem import CMPState, ConstrainedMinimizationProblem
from cooper.utils import StateLogger

from . import formulation, optim, utils
