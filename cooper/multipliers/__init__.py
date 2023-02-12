from typing import Union

from .multipliers import ConstantMultiplier, DenseMultiplier, ExplicitMultiplier, ImplicitMultiplier, SparseMultiplier

MULTIPLIER_TYPE = Union[DenseMultiplier, SparseMultiplier, ImplicitMultiplier, ConstantMultiplier]
