import pytest
import torch

import cooper
import cooper.optim.optimizer


@pytest.fixture
def primal_params():
    return torch.rand(1, requires_grad=True, generator=torch.Generator().manual_seed(0))


@pytest.fixture(
    params=[
        cooper.optim.UnconstrainedOptimizer,
        cooper.optim.AlternatingDualPrimalOptimizer,
        cooper.optim.AlternatingPrimalDualOptimizer,
        cooper.optim.SimultaneousOptimizer,
    ]
)
def cooper_optimizer_class(request):
    return request.param


@pytest.fixture
def cooper_optimizer(cooper_optimizer_class, cmp_instance, primal_params):
    primal_optimizer = torch.optim.SGD([primal_params], lr=0.1)
    dual_optimizer = None
    if cooper_optimizer_class != cooper.optim.UnconstrainedOptimizer:
        dual_optimizer = torch.optim.SGD(cmp_instance.dual_parameters(), lr=0.1, maximize=True)
    optimizer = cooper_optimizer_class(cmp_instance, primal_optimizer, dual_optimizer)
    return optimizer


def test_load_state_dict_mismatch_primal(cooper_optimizer, cooper_optimizer_class, cmp_instance, primal_params):
    state = cooper_optimizer.state_dict()
    new_optimizer = cooper_optimizer_class(
        cmp_instance, [torch.optim.SGD([primal_params], lr=0.1) for _ in range(2)], cooper_optimizer.dual_optimizers
    )

    with pytest.raises(
        ValueError, match=r"The number of primal optimizers does not match the number of primal optimizer states."
    ):
        new_optimizer.load_state_dict(state)


def test_load_state_dict_mismatch_dual(cooper_optimizer, cooper_optimizer_class):
    if cooper_optimizer_class == cooper.optim.UnconstrainedOptimizer:
        pytest.skip("UnconstrainedOptimizer does not have dual optimizers.")

    state = cooper_optimizer.state_dict()
    new_optimizer = cooper.optim.UnconstrainedOptimizer(cooper_optimizer.cmp, cooper_optimizer.primal_optimizers)

    with pytest.raises(
        ValueError, match=r"Optimizer state dict contains ``dual_optimizer_states`` but ``dual_optimizers`` is None."
    ):
        new_optimizer.load_state_dict(state)


def test_load_state_dict_mismatch_dual_count(cooper_optimizer, cooper_optimizer_class, cmp_instance):
    if cooper_optimizer_class == cooper.optim.UnconstrainedOptimizer:
        pytest.skip("UnconstrainedOptimizer does not have dual optimizers.")

    state = cooper_optimizer.state_dict()
    new_optimizer = cooper_optimizer_class(
        cmp_instance,
        cooper_optimizer.primal_optimizers,
        [
            torch.optim.SGD([torch.ones(1, requires_grad=True)], lr=0.1, maximize=True),
            torch.optim.SGD([torch.ones(1, requires_grad=True)], lr=0.1, maximize=True),
        ],
    )

    with pytest.raises(
        ValueError, match=r"The number of dual optimizers does not match the number of dual optimizer states."
    ):
        new_optimizer.load_state_dict(state)
