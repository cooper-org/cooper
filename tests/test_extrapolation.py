#!/usr/bin/env python

"""Tests for Extrapolation optimizers."""

# Import basic closure example from helpers
import cooper_test_utils
import pytest
import torch

import cooper


def problem_data(aim_device, primal_optim_cls):

    test_problem_data = cooper_test_utils.build_test_problem(
        aim_device=aim_device,
        primal_optim_cls=primal_optim_cls,
        primal_init=[0.0, -1.0],
        dual_optim_cls=cooper.optim.ExtraSGD,
        use_ineq=True,
        use_proxy_ineq=False,
        dual_restarts=False,
        alternating=False,
    )

    # params, cmp, coop, formulation, device, mktensor
    return test_problem_data.as_tuple()


@pytest.mark.parametrize("aim_device", ["cpu", "cuda"])
@pytest.mark.parametrize("primal_optimizer_cls", [cooper.optim.ExtraSGD, cooper.optim.ExtraAdam])
def test_extrapolation(aim_device, primal_optimizer_cls):
    """ """

    params, cmp, coop, formulation, device, _ = problem_data(aim_device, primal_optimizer_cls)

    for step_id in range(2000):
        coop.zero_grad()
        lagrangian = formulation.compute_lagrangian(cmp.closure, params)
        formulation.backward(lagrangian)
        coop.step(cmp.closure, params)

    if device == "cuda":
        assert cmp.state.loss.is_cuda
        assert cmp.state.eq_defect is None or cmp.state.eq_defect.is_cuda
        assert cmp.state.ineq_defect is None or cmp.state.ineq_defect.is_cuda

    # TODO: Why do we need such relaxed tolerance for this test to pass?
    if primal_optimizer_cls == cooper.optim.ExtraSGD:
        atol = 1e-8
    else:
        atol = 1e-3
    assert torch.allclose(params[0], torch.tensor(2.0 / 3.0), atol=atol)
    assert torch.allclose(params[1], torch.tensor(1.0 / 3.0), atol=atol)


@pytest.mark.parametrize("aim_device", ["cpu", "cuda"])
@pytest.mark.parametrize("primal_optimizer_cls", [cooper.optim.ExtraSGD])
def test_manual_extrapolation(aim_device, primal_optimizer_cls):
    """ """

    params, cmp, coop, formulation, device, mktensor = problem_data(aim_device, primal_optimizer_cls)

    coop.zero_grad()
    lagrangian = formulation.compute_lagrangian(cmp.closure, params)

    # Check loss, proxy and non-proxy defects after forward pass
    assert torch.allclose(lagrangian, mktensor(2.0))
    assert torch.allclose(cmp.state.loss, mktensor(2.0))
    assert torch.allclose(cmp.state.ineq_defect, mktensor([2.0, -2.0]))
    assert cmp.state.eq_defect is None

    # Multiplier initialization
    assert torch.allclose(formulation.state()[0], mktensor([0.0, 0.0]))
    assert formulation.state()[1] is None

    # Check primal and dual gradients after backward. Dual gradient must match
    # ineq_defect
    formulation.backward(lagrangian)
    assert torch.allclose(params.grad, mktensor([0.0, -4.0]))
    assert torch.allclose(formulation.state()[0].grad, cmp.state.ineq_defect)

    # Check updated primal and dual variable values
    coop.step(cmp.closure, params)
    assert torch.allclose(params, mktensor([2.0e-4, -0.9614]))
    assert torch.allclose(formulation.state()[0], mktensor([0.0196, 0.0]))
