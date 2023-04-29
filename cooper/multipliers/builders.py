from typing import Optional

import torch

from cooper.constraints.constraint_state import ConstraintType

from .multipliers import DenseMultiplier, ExplicitMultiplier, IndexedMultiplier


def build_explicit_multiplier(
    constraint_type: ConstraintType,
    shape: int,
    device: Optional[torch.device] = None,
    restart_on_feasible: bool = False,
    is_indexed: bool = False,
    dtype: torch.dtype = torch.float32,
) -> ExplicitMultiplier:
    """Initializes a dense or sparse multiplier at zero, with desired shape, dtype and
    destination device."""

    multiplier_class = IndexedMultiplier if is_indexed else DenseMultiplier
    enforce_positive = constraint_type == ConstraintType.INEQUALITY

    tensor_factory = dict(dtype=dtype, device=device)
    # Indexed multipliers require the weight to be 2D
    tensor_factory["size"] = (shape, 1) if is_indexed else (shape,)

    multiplier_kwargs = dict(enforce_positive=enforce_positive, restart_on_feasible=restart_on_feasible)

    return multiplier_class(init=torch.zeros(**tensor_factory), **multiplier_kwargs)
