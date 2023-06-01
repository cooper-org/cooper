from typing import Union

from .builders import build_explicit_multiplier
from .multipliers import DenseMultiplier, ExplicitMultiplier, ImplicitMultiplier, IndexedMultiplier
from .penalty_coefficients import PenaltyCoefficient, IndexedPenaltyCoefficient

MULTIPLIER_TYPE = Union[DenseMultiplier, IndexedMultiplier, ImplicitMultiplier]
