# create class that inherits from cooper.multipliers.MultiplierModel
import cooper
import torch
import abc


def test_multiplier_model_init():
    # create a model
    model = torch.nn.Linear(10, 1)
    # create a multiplier model
    multiplier_model = cooper.multipliers.MultiplierModel(model)
    # check that the multiplier model is a MultiplierModel
    assert isinstance(multiplier_model, cooper.multipliers.BaseMultiplier)


def test_multiplier_model_forward():
    # create a model
    model = torch.nn.Linear(10, 1)
    # create a multiplier model
    multiplier_model = cooper.multipliers.MultiplierModel(model)
    # create a tensor of constraint features
    constraint_features = torch.randn(100, 10)
    # check that the multiplier model is a MultiplierModel
    assert torch.allclose(
        multiplier_model(constraint_features), model(constraint_features)
    )
