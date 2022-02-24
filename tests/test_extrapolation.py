#!/usr/bin/env python

"""Tests for Extrapolation optimizers."""

import functools

# Import basic closure example from helpers
import const_min_problem_2d
import pytest
import testing_utils
import torch

import cooper


@pytest.mark.parametrize("aim_device", ["cpu", "cuda"])
@pytest.mark.parametrize("primal_optimizer_str", ["ExtraSGD", "ExtraAdam"])
def test_extrapolation(aim_device, primal_optimizer_str):
    """
    Simple test on a bi-variate quadratic programming problem
        min x**2 + 2*y**2
        st.
            x + y >= 1
            x**2 + y <= 1

    Verified solution from WolframAlpha (x=2/3, y=1/3)
    Link to WolframAlpha query: https://tinyurl.com/ye8dw6t3
    """

    device, skip = testing_utils.get_device_skip(aim_device, torch.cuda.is_available())

    if skip.do_skip:
        pytest.skip(skip.skip_reason)

    params = torch.nn.Parameter(torch.tensor([0.0, -1.0], device=device))
    primal_optimizer = getattr(cooper.optim, primal_optimizer_str)([params], lr=1e-2)
    dual_optimizer = cooper.optim.partial(cooper.optim.ExtraSGD, lr=1e-2)

    cmp = const_min_problem_2d.CustomCMP(is_constrained=True)
    formulation = cooper.LagrangianFormulation(cmp)

    coop = cooper.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optimizer,
        dual_optimizer=dual_optimizer,
        dual_restarts=False,
    )

    for _ in range(2000):
        coop.zero_grad()
        lagrangian = formulation.composite_objective(params, use_ineq=True)
        formulation.custom_backward(lagrangian)
        coop.step(params, use_ineq=True)

    if device == "cuda":
        assert cmp.loss.is_cuda
        assert cmp.eq_defect is None or cmp.eq_defect.is_cuda
        assert cmp.ineq_defect is None or cmp.ineq_defect.is_cuda

    # TODO: Why do we need such relatex tolerance for this test to pass?
    if primal_optimizer == "ExtraSGD":
        atol = 1e-8
    else:
        atol = 1e-3
    assert torch.allclose(params[0], torch.tensor(2.0 / 3.0), atol=atol)
    assert torch.allclose(params[1], torch.tensor(1.0 / 3.0), atol=atol)


@pytest.mark.parametrize("aim_device", ["cpu", "cuda"])
@pytest.mark.parametrize("primal_optimizer", ["ExtraSGD"])
def test_manual_extrapolation(aim_device, primal_optimizer):
    """
    Simple test on a bi-variate quadratic programming problem
        min x**2 + 2*y**2
        st.
            x + y >= 1
            x**2 + y <= 1

    Verified solution from WolframAlpha (x=2/3, y=1/3)
    Link to WolframAlpha query: https://tinyurl.com/ye8dw6t3
    """

    device, skip = testing_utils.get_device_skip(aim_device, torch.cuda.is_available())

    if skip.do_skip:
        pytest.skip(skip.skip_reason)

    params = torch.nn.Parameter(torch.tensor([0.0, -1.0], device=device))
    primal_optimizer = getattr(cooper.optim, primal_optimizer)([params], lr=1e-2)
    dual_optimizer = cooper.optim.partial(cooper.optim.ExtraSGD, lr=1e-2)

    cmp = const_min_problem_2d.CustomCMP(is_constrained=True)
    formulation = cooper.LagrangianFormulation(cmp)

    coop = cooper.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optimizer,
        dual_optimizer=dual_optimizer,
        dual_restarts=False,
    )

    # Helper function to instantiate tensors in correct device
    mktensor = functools.partial(torch.tensor, device=device)

    coop.zero_grad()
    lagrangian = formulation.composite_objective(params, use_ineq=True)

    # Check loss, proxy and non-proxy defects after forward pass
    assert torch.allclose(lagrangian, mktensor(2.0))
    assert torch.allclose(cmp.loss, mktensor(2.0))
    assert torch.allclose(cmp.ineq_defect, mktensor([2.0, -2.0]))
    assert cmp.eq_defect is None

    # Multiplier initialization
    assert torch.allclose(formulation.state()[0], mktensor([0.0, 0.0]))
    assert formulation.state()[1] is None

    # Check primal and dual gradients after backward. Dual gradient must match
    # ineq_defect
    formulation.custom_backward(lagrangian)
    assert torch.allclose(params.grad, mktensor([0.0, -4.0]))
    assert torch.allclose(formulation.state()[0].grad, cmp.ineq_defect)

    # Check updated primal and dual variable values
    coop.step(params, use_ineq=True)
    assert torch.allclose(params, mktensor([2.0e-4, -0.9614]))
    assert torch.allclose(formulation.state()[0], mktensor([0.0196, 0.0]))
