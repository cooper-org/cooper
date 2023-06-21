"""Top-level package for Cooper."""

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

    warnings.warn("Could not retrieve Cooper version!")

from cooper.cmp import CMPState, ConstrainedMinimizationProblem, LagrangianStore
from cooper.constraints import ConstraintGroup, ConstraintState, ConstraintType
from cooper.formulations import FormulationType

from . import multipliers, optim, utils
