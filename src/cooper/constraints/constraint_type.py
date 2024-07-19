from enum import Enum, auto


# TODO(gallego-posada): Maybe we can move this to the `constraint` module. Need to
#  ensure no circular imports are created.
class ConstraintType(Enum):
    """Constraint type enumeration."""

    EQUALITY = auto()
    INEQUALITY = auto()
