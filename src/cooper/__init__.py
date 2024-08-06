"""Top-level package for Cooper."""

from importlib.metadata import PackageNotFoundError, version

from . import formulations, multipliers, optim, utils
from .cmp import CMPState, ConstrainedMinimizationProblem, LagrangianStore
from .constraints import Constraint, ConstraintState
from .formulations import AugmentedLagrangianFormulation, Formulation, LagrangianFormulation
from .utils import ConstraintType

try:
    __version__ = version("cooper-optim")
except PackageNotFoundError:
    # package is not installed
    import warnings

    warnings.warn("Could not retrieve Cooper version!")
    del warnings
del version, PackageNotFoundError
