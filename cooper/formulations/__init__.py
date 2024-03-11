from enum import Enum

from .formulations import AugmentedLagrangianFormulation, Formulation, LagrangianFormulation


class FormulationType(Enum):
    LAGRANGIAN = LagrangianFormulation
    AUGMENTED_LAGRANGIAN = AugmentedLagrangianFormulation
