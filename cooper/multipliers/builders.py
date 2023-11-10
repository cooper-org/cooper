from typing import Optional

import torch

from cooper.constraints.constraint_state import ConstraintType

from .multipliers import DenseMultiplier, ExplicitMultiplier, IndexedMultiplier


def build_explicit_multiplier(
    constraint_type: ConstraintType,
    num_constraints: int,
    restart_on_feasible: bool = False,
    is_indexed: bool = False,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> ExplicitMultiplier:
    """Initializes a dense or indexed multiplier at zero, given a desired shape, dtype
    and device."""

    if constraint_type == ConstraintType.PENALTY:
        raise ValueError("`Penalty` constraints do not admit multipliers.")
    elif constraint_type not in [ConstraintType.EQUALITY, ConstraintType.INEQUALITY]:
        raise ValueError(f"Constraint type {constraint_type} not recognized.")

    multiplier_class = IndexedMultiplier if is_indexed else DenseMultiplier
    enforce_positive = constraint_type == ConstraintType.INEQUALITY

    tensor_factory = dict(dtype=dtype, device=device)
    # Indexed multipliers require the weight to be 2D
    tensor_factory["size"] = (num_constraints, 1) if is_indexed else (num_constraints,)

    multiplier = multiplier_class(
        init=torch.zeros(**tensor_factory), enforce_positive=enforce_positive, restart_on_feasible=restart_on_feasible
    )

    return multiplier
