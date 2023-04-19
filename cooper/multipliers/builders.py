import torch

from .multipliers import DenseMultiplier, ExplicitMultiplier, SparseMultiplier


def build_explicit_multiplier(
    constraint_type: str, shape: int, dtype: torch.dtype, device: torch.device, is_sparse: bool
) -> ExplicitMultiplier:
    """Initializes a dense or sparse multiplier at zero, with desired shape, dtype and
    destination device."""

    multiplier_class = SparseMultiplier if is_sparse else DenseMultiplier
    enforce_positive = constraint_type == "ineq"

    tensor_factory = dict(dtype=dtype, device=device)
    tensor_factory["size"] = (shape, 1) if is_sparse else (shape,)

    return multiplier_class(init=torch.zeros(**tensor_factory), enforce_positive=enforce_positive)
