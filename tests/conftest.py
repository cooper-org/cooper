import pytest
import torch

from tests.helpers.cooper_test_utils import (  # noqa:  F401
    Toy2dCMP_params_init,
    Toy2dCMP_problem_properties,
    use_multiple_primal_optimizers,
)


@pytest.fixture(scope="session", params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("Aim device 'cuda' is not available.")

    return torch.device(device)
