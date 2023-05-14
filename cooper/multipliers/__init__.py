from typing import Union

from .builders import build_explicit_multiplier
from .multipliers import DenseMultiplier, ExplicitMultiplier, ImplicitMultiplier, IndexedMultiplier
from .penalties import PenaltyCoefficient

MULTIPLIER_TYPE = Union[DenseMultiplier, IndexedMultiplier, ImplicitMultiplier]
