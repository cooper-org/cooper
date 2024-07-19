from collections.abc import Sequence
from typing import TypeVar, Union

T = TypeVar("T")
OneOrSequence = Union[T, Sequence[T]]
