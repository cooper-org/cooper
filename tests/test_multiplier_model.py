# create class that inherits from cooper.multipliers.MultiplierModel
import cooper
import torch

class ToyMultiplierModel(cooper.multipliers.MultiplierModel):
    """
    Simplest MultiplierModel possible, a linear model with a single output.
    """

    def __init__(self, n_params, n_hidden_units, device):
        super().__init__()
        self.linear1 = torch.nn.Linear(n_params, n_hidden_units, device=device)
        self.linear2 = torch.nn.Linear(n_hidden_units, 1, device=device)

    def forward(self, constraint_features: torch.Tensor):
        x = self.linear1(constraint_features)
        x = torch.relu(x)
        x = self.linear2(x)
        return torch.reshape(torch.nn.functional.relu(x), (-1,))

def test_multiplier_model_init():

    multiplier_model = ToyMultiplierModel(10, 10, "cpu")

    assert isinstance(multiplier_model, cooper.multipliers.BaseMultiplier)
    assert isinstance(multiplier_model, cooper.multipliers.MultiplierModel)
