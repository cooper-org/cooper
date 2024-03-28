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


def check_save_load_state_dict(multiplier, explicit_multiplier_class, multiplier_shape, all_indices, random_seed):
    generator = testing_utils.frozen_rand_generator(random_seed)

    multiplier_init = torch.randn(*multiplier_shape, generator=generator)
    if multiplier.enforce_positive:
        multiplier_init = multiplier_init.relu()
    new_multiplier = explicit_multiplier_class(init=multiplier_init)
    new_multiplier.enforce_positive = multiplier.enforce_positive

    # Save to file to force reading from file so we can ensure correct loading
    with tempfile.TemporaryDirectory() as tmpdirname:
        torch.save(multiplier.state_dict(), os.path.join(tmpdirname, "multiplier.pt"))
        state_dict = torch.load(os.path.join(tmpdirname, "multiplier.pt"))

    new_multiplier.load_state_dict(state_dict)

    if isinstance(multiplier, cooper.multipliers.IndexedMultiplier):
        assert torch.allclose(multiplier(all_indices), new_multiplier(all_indices))
    else:
        assert torch.allclose(multiplier(), new_multiplier())
