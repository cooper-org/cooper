from enum import Enum

from .formulations import (
    AugmentedLagrangianFormulation,
    Formulation,
    LagrangianFormulation,
    PenaltyFormulation,
    QuadraticPenaltyFormulation,
)


class FormulationType(Enum):
    PENALTY = PenaltyFormulation
    QUADRATIC_PENALTY = QuadraticPenaltyFormulation
    LAGRANGIAN = LagrangianFormulation
    AUGMENTED_LAGRANGIAN = AugmentedLagrangianFormulation
