from enum import Enum

from .formulations import (
    AugmentedLagrangianFormulation,
    Formulation,
    LagrangianFormulation,
    PenaltyFormulation,
    QuadraticPenaltyFormulation,
)
from .utils import extract_and_patch_violations


class FormulationType(Enum):
    PENALTY = PenaltyFormulation
    QUADRATIC_PENALTY = QuadraticPenaltyFormulation
    LAGRANGIAN = LagrangianFormulation
    AUGMENTED_LAGRANGIAN = AugmentedLagrangianFormulation
