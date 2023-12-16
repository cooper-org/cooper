import os
import sys

import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "helpers"))

import testing_utils  # noqa: E402
from cooper_test_utils import (  # noqa: E402, F401
    Toy2dCMP_params_init,
    Toy2dCMP_problem_properties,
    use_multiple_primal_optimizers,
)


@pytest.fixture(scope="session", params=["cpu", "cuda"])
def device(request):
    device, skip, skip_reason = testing_utils.get_device_or_skip(request.param, torch.cuda.is_available())
    if skip:
        pytest.skip(skip_reason)

    return device
