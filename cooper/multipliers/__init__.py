from typing import Union

from .builders import build_explicit_multiplier
from .multipliers import ConstantMultiplier, DenseMultiplier, ExplicitMultiplier, ImplicitMultiplier, SparseMultiplier

MULTIPLIER_TYPE = Union[DenseMultiplier, SparseMultiplier, ImplicitMultiplier, ConstantMultiplier]
