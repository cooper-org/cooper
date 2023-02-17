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

# from cooper.optim import (
#     AlternatingConstrainedOptimizer,
#     ConstrainedOptimizer,
#     CooperOptimizer,
#     ExtrapolationConstrainedOptimizer,
#     SimultaneousConstrainedOptimizer,
#     UnconstrainedOptimizer,
# )

from cooper.cmp import CMPState, ConstrainedMinimizationProblem
from cooper.constraints import ConstraintGroup, ConstraintState

from . import multipliers, utils  # , optim
