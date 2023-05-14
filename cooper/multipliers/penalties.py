import torch


class PenaltyCoefficient:
    """Constant (non-trainable) coefficient class used for penalized formulations.

    Args:
        init: Value of the penalty.
    """

    def __init__(self, init: torch.Tensor):
        if init.requires_grad:
            raise ValueError("PenaltyCoefficient should not be trainable.")
        self.weight = init
        self.device = init.device

    def update_value_(self, value: torch.Tensor):
        """Update the value of the penalty."""
        self.weight.data = value

    def __call__(self):
        """Return the current value of the multiplier."""
        return torch.clone(self.weight)

    def parameters(self):
        """Return an empty iterator for consistency with multipliers which are
        :py:class:`~torch.nn.Module`."""
        return iter(())

    def state_dict(self):
        return {"weight": self.weight}

    def load_state_dict(self, state_dict):
        self.weight = state_dict["weight"]

    def __repr__(self):
        return f"PenaltyCoefficient({self.weight})"
