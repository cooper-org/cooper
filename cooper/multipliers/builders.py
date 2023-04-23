import torch

from .multipliers import DenseMultiplier, ExplicitMultiplier, IndexedMultiplier


def build_explicit_multiplier(
    constraint_type: str,
    shape: int,
    device: torch.device,
    restart_on_feasible: bool = False,
    is_indexed: bool = False,
    dtype: torch.dtype = torch.float32,
) -> ExplicitMultiplier:
    """Initializes a dense or sparse multiplier at zero, with desired shape, dtype and
    destination device."""

    multiplier_class = IndexedMultiplier if is_indexed else DenseMultiplier
    enforce_positive = constraint_type == "ineq"

    tensor_factory = dict(dtype=dtype, device=device)
    # Indexed multipliers require the weight to be 2D
    tensor_factory["size"] = (shape, 1) if is_indexed else (shape,)

    return multiplier_class(
        init=torch.zeros(**tensor_factory), enforce_positive=enforce_positive, restart_on_feasible=restart_on_feasible
    )
