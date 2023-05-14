from typing import Union

from .builders import build_explicit_multiplier
from .penalties import PenaltyCoefficient

MULTIPLIER_TYPE = Union[DenseMultiplier, IndexedMultiplier, ImplicitMultiplier, ConstantMultiplier]
