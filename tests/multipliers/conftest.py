import pytest
import torch

import cooper
from tests.helpers import testing_utils


@pytest.fixture
def random_seed():
    return 1516516984916


@pytest.fixture(params=[cooper.ConstraintType.EQUALITY, cooper.ConstraintType.INEQUALITY])
def constraint_type(request):
    return request.param


@pytest.fixture(params=[(100,), (1000,)])
def multiplier_shape(request):
    return request.param


@pytest.fixture(params=[cooper.multipliers.DenseMultiplier, cooper.multipliers.IndexedMultiplier])
def multiplier_class(request):
    return request.param


@pytest.fixture
def init_multiplier_tensor(constraint_type, multiplier_shape, random_seed):
    generator = testing_utils.frozen_rand_generator(random_seed)
    raw_init = torch.randn(*multiplier_shape, generator=generator)
    if constraint_type == cooper.ConstraintType.INEQUALITY:
        return raw_init.relu()
    else:
        return raw_init


@pytest.fixture
def all_indices(multiplier_shape):
    return torch.arange(0, multiplier_shape[0], dtype=torch.long)
