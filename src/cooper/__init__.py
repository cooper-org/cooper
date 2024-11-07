"""Top-level package for Cooper."""

from importlib.metadata import PackageNotFoundError, version

from cooper.cmp import CMPState, ConstrainedMinimizationProblem, LagrangianStore
from cooper.constraints import Constraint, ConstraintState
from cooper.formulations import (
    AugmentedLagrangianFormulation,
    AugmentedLagrangianMethodFormulation,
    Formulation,
    LagrangianFormulation,
)
from cooper.utils import ConstraintType

from . import formulations, multipliers, optim, utils

try:
    __version__ = version("cooper-optim")
except PackageNotFoundError:
    # package is not installed
    import warnings

    warnings.warn("Could not retrieve Cooper version!")
    del warnings
del version, PackageNotFoundError
