from collections.abc import Sequence
from typing import Optional

from cooper.utils.annotations import OneOrSequence, T


def ensure_sequence(argument: Optional[OneOrSequence[T]]) -> Optional[Sequence[T]]:
    """Ensures that an argument is an instance of Sequence by wrapping it into a list
    whenever necessary. When argument is None, None is returned without wrapping.
    """
    if argument is None:
        return None
    return argument if isinstance(argument, Sequence) else [argument]
