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

from cooper.augmented_lagrangian import AugmentedLagrangianFormulation
from cooper.constrained_optimizer import ConstrainedOptimizer
from cooper.lagrangian_formulation import LagrangianFormulation
from cooper.problem import (
    CMPState,
    ConstrainedMinimizationProblem,
    UnconstrainedFormulation,
)
from cooper.state_logger import StateLogger

from . import optim, utils
