import pytest
import torch

import cooper
import testing


@pytest.fixture
def random_seed():
    return 1516516984916


@pytest.fixture(params=[cooper.ConstraintType.EQUALITY, cooper.ConstraintType.INEQUALITY])
def constraint_type(request):
    return request.param


@pytest.fixture(params=[1, 100, 1000])
def num_constraints(request):
    return request.param


@pytest.fixture(params=[cooper.multipliers.DenseMultiplier, cooper.multipliers.IndexedMultiplier])
def multiplier_class(request):
    return request.param


@pytest.fixture
def init_multiplier_tensor(constraint_type, num_constraints, random_seed):
    generator = testing.frozen_rand_generator(random_seed)
    raw_init = torch.randn(num_constraints, generator=generator)
    if constraint_type == cooper.ConstraintType.INEQUALITY:
        return raw_init.relu()
    return raw_init


@pytest.fixture
def all_indices(num_constraints):
    return torch.arange(num_constraints, dtype=torch.long)
