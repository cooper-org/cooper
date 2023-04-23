import tempfile

import pytest
import torch

from cooper import multipliers
from tests.helpers import testing_utils


@pytest.fixture
def random_seed():
    return 1516516984916


@pytest.fixture(params=[(100,), (1000,)])
def multiplier_shape(request):
    return request.param


@pytest.fixture
def _init_tensor(multiplier_shape, random_seed):
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

    init = torch.randn(*multiplier_shape, generator=generator)
    if explicit_multiplier_class == multipliers.IndexedMultiplier:
        init = init.unsqueeze(1)
    new_multiplier = explicit_multiplier_class(init=init)

    # Save to file to force reading from file so we can ensure correct loading
    tmp = tempfile.NamedTemporaryFile()
    torch.save(multiplier.state_dict(), tmp.name)
    state_dict = torch.load(tmp.name)
    tmp.close()

    new_multiplier.load_state_dict(state_dict)

    assert multiplier.implicit_constraint_type == new_multiplier.implicit_constraint_type
    assert multiplier.restart_on_feasible == new_multiplier.restart_on_feasible

    if isinstance(multiplier, multipliers.IndexedMultiplier):
        assert torch.allclose(multiplier(all_indices), new_multiplier(all_indices))
    else:
        assert torch.allclose(multiplier(), new_multiplier())
