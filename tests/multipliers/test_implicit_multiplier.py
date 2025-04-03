# Copyright (C) 2025 The Cooper Developers.
# Licensed under the MIT License.

import pytest

import cooper


def test_implicit_multiplier_forward_raises():
    class TestImplicitMultiplier(cooper.multipliers.ImplicitMultiplier):
        pass

    with pytest.raises(TypeError):
        _ = TestImplicitMultiplier()
