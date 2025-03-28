from collections.abc import Sequence
from enum import Enum, auto
from typing import TypeVar, Union

T = TypeVar("T")
OneOrSequence = Union[T, Sequence[T]]


# TODO(gallego-posada): Maybe we can move this to the `constraint` module. Need to
#  ensure no circular imports are created.
class ConstraintType(Enum):
    """A constraint type is one of: ``EQUALITY`` or ``INEQUALITY``."""

    EQUALITY = auto()
    INEQUALITY = auto()
