import os
import tempfile

import pytest
import torch

import cooper
from tests.helpers import testing_utils


@pytest.fixture
def random_seed():
    return 1516516984916


@pytest.fixture(params=[(100,), (1000,)])
def multiplier_shape(request):
    return request.param


@pytest.fixture(params=[cooper.multipliers.DenseMultiplier, cooper.multipliers.IndexedMultiplier])
def multiplier_class(request):
    return request.param


@pytest.fixture
def init_tensor(multiplier_shape, random_seed):
    generator = testing_utils.frozen_rand_generator(random_seed)
    return torch.randn(*multiplier_shape, generator=generator)


@pytest.fixture
def all_indices(multiplier_shape):
    return torch.arange(0, multiplier_shape[0], dtype=torch.long)


@pytest.fixture
def feasible_indices(multiplier_shape, random_seed):
    generator = testing_utils.frozen_rand_generator(random_seed)
    return torch.randint(0, 2, multiplier_shape, dtype=torch.bool, generator=generator)


def check_save_load_state_dict(multiplier, explicit_multiplier_class, multiplier_shape, all_indices, random_seed):
    generator = testing_utils.frozen_rand_generator(random_seed)

    multiplier_init = torch.randn(*multiplier_shape, generator=generator)
    if multiplier.constraint_type == cooper.ConstraintType.INEQUALITY:
        multiplier_init = multiplier_init.relu()
    new_multiplier = explicit_multiplier_class(constraint_type=multiplier.constraint_type, init=multiplier_init)

    # Save to file to force reading from file so we can ensure correct loading
    with tempfile.TemporaryDirectory() as tmpdirname:
        torch.save(multiplier.state_dict(), os.path.join(tmpdirname, "multiplier.pt"))
        state_dict = torch.load(os.path.join(tmpdirname, "multiplier.pt"))

    new_multiplier.load_state_dict(state_dict)

    assert multiplier.constraint_type == new_multiplier.constraint_type
    assert multiplier.restart_on_feasible == new_multiplier.restart_on_feasible

    if isinstance(multiplier, cooper.multipliers.IndexedMultiplier):
        assert torch.allclose(multiplier(all_indices), new_multiplier(all_indices))
    else:
        assert torch.allclose(multiplier(), new_multiplier())
