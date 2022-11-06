# create class that inherits from cooper.multipliers.MultiplierModel
import cooper
import torch
import abc

from cooper.multipliers import BaseMultiplier


class MultiplierModel(BaseMultiplier):
    def __init__(self, model: torch.nn.Module, is_positive: bool = False):
        super().__init__()
        self.model = model
        self.is_positive = is_positive
        self.device = next(self.model.parameters()).device

    def grad(self):
        """Yields the current gradients stored in each fo the model parameters."""
        for param in self.model.parameters():
            if param.grad is not None:
                yield param.grad

    def forward(self, constraint_features: torch.Tensor):
        return self.model(constraint_features)

    def shape(self):
        """
        Returns the shape of the explicit multipliers. In the case of implicit
        multipliers, this should return the *actual* predicted multipliers.
        """
        pass

    def project_(self):
        raise RuntimeError("""project_ method does not exist for MultiplierModel.""")

    def restart_if_feasible_(self):
        raise RuntimeError(
            """restart_if_feasible_ method does not exist for MultiplierModel."""
        )


def test_multiplier_model_init():
    # create a model
    model = torch.nn.Linear(10, 1)
    # create a multiplier model
    multiplier_model = MultiplierModel(model)
    # check that the multiplier model is a MultiplierModel
    assert isinstance(multiplier_model, cooper.multipliers.BaseMultiplier)


def test_multiplier_model_forward():
    # create a model
    model = torch.nn.Linear(10, 1)
    # create a multiplier model
    multiplier_model = MultiplierModel(model)
    # create a tensor of constraint features
    constraint_features = torch.randn(100, 10)
    # check that the multiplier model is a MultiplierModel
    assert torch.allclose(
        multiplier_model(constraint_features), model(constraint_features)
    )