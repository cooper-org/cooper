#!/usr/bin/env python

"""Tests for SimultaneousConstrainedOptimizer class."""

import cooper_test_utils
import pytest
import testing_utils
import torch

from cooper import UnconstrainedOptimizer


@pytest.fixture
def toy_problem():
    return cooper_test_utils.Toy2dCMP(use_ineq_constraints=False, use_constraint_surrogate=False)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):

    device, skip = testing_utils.get_device_skip(request.param, torch.cuda.is_available())
    if skip.do_skip:
        pytest.skip(skip.reason)

    return device


@pytest.fixture
def params(device):
    param_init = [torch.tensor(0.0), torch.tensor(-1.0)]
    return cooper_test_utils.build_params_from_init(param_init, device)


@pytest.fixture(params=[True, False])
def multiple_optimizers(request):
    return request.param


@pytest.fixture
def primal_optimizers(multiple_optimizers, params):
    if multiple_optimizers:
        return [torch.optim.SGD([params[0]], lr=1e-2, momentum=0.3), torch.optim.Adam([params[1]], lr=1e-2)]
    else:
        return torch.optim.SGD(params, lr=1e-2, momentum=0.3)


@pytest.fixture
def unconstrained_optimizer(primal_optimizers):
    return UnconstrainedOptimizer(primal_optimizers)


def test_unconstrained_training(params, toy_problem, unconstrained_optimizer):

    for step_id in range(1500):
        unconstrained_optimizer.zero_grad()

        cmp_state = toy_problem.closure(params)
        lagrangian, multipliers = cmp_state.populate_lagrangian(return_multipliers=True)

        # When using the unconstrained formulation, lagrangian = loss
        assert torch.allclose(lagrangian, cmp_state.loss)
        # There are no multipliers in the unconstrained case
        assert len(multipliers) == 0

        cmp_state.backward()
        unconstrained_optimizer.step()

    # This unconstrained quadratic form has minimum at the origin
    assert torch.allclose(params[0], torch.tensor(0.0))
    assert torch.allclose(params[1], torch.tensor(0.0))
