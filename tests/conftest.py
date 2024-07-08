import pytest
import torch


@pytest.fixture(scope="session", params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("Aim device 'cuda' is not available.")
    return torch.device(request.param)
