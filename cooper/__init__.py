"""Top-level package for Constrained Optimization in Pytorch."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cooper")
except PackageNotFoundError:
    # package is not installed
    import warnings

    warnings.warn("Could not retrieve cooper version!")

from cooper.constrained_optimizer import ConstrainedOptimizer
from cooper.lagrangian_formulation import LagrangianFormulation
from cooper.problem import CMPState, ConstrainedMinimizationProblem
from cooper.state_logger import StateLogger

from . import optim
