from collections.abc import Sequence
from typing import Any


def ensure_sequence(argument: Any):
    """
    Ensures that an argument is an instance of Sequence by wrapping it into a list
    whenever necessary. When argument is None, None is returned without wrapping.
    """
    return argument if isinstance(argument, Sequence) else [argument]
