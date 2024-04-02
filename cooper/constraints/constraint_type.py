from enum import Enum, auto


class ConstraintType(Enum):
    """Constraint type enumeration."""

    EQUALITY = auto()
    INEQUALITY = auto()
