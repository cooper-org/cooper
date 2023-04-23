from typing import Union

from .builders import build_explicit_multiplier
from .multipliers import ConstantMultiplier, DenseMultiplier, ExplicitMultiplier, ImplicitMultiplier, IndexedMultiplier

MULTIPLIER_TYPE = Union[DenseMultiplier, IndexedMultiplier, ImplicitMultiplier, ConstantMultiplier]
