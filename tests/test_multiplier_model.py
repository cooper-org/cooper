import cooper


def test_multiplier_model_init():
    class DummyMultiplierModel(cooper.multipliers.MultiplierModel):
        def __init__(self):
            super().__init__()

        def forward(self):
            pass

    multiplier_model = DummyMultiplierModel()

    assert isinstance(multiplier_model, cooper.multipliers.BaseMultiplier)
    assert isinstance(multiplier_model, cooper.multipliers.MultiplierModel)
