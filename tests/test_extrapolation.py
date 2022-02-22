#!/usr/bin/env python

"""Tests for Extrapolation optimizers."""

import functools

# Import basic closure example from helpers
import closure_2d
import pytest
import testing_utils
import torch

import torch_coop


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
    primal_optimizer = getattr(torch_coop.optim, primal_optimizer_str)(
        [params], lr=1e-2
    )

    dual_optimizer = torch_coop.optim.partial(torch_coop.optim.ExtraSGD, lr=1e-2)

    construct_closure = functools.partial(closure_2d.construct_closure, use_ineq=True)

    cmp = torch_coop.ConstrainedMinimizationProblem(is_constrained=True)
    formulation = torch_coop.LagrangianFormulation(cmp)

    coop = torch_coop.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optimizer,
        dual_optimizer=dual_optimizer,
        dual_restarts=False,
    )

    for step_id in range(2000):
        coop.zero_grad()

        closure = construct_closure(params)
        lagrangian = coop.composite_objective(closure)

        coop.custom_backward(lagrangian)

        coop.step(construct_closure, closure_args=params)

    if device == "cuda":
        assert cmp.state.loss.is_cuda
        assert cmp.state.eq_defect is None or cmp.state.eq_defect.is_cuda
        assert cmp.state.ineq_defect is None or cmp.state.ineq_defect.is_cuda

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
    primal_optimizer = getattr(torch_coop.optim, primal_optimizer)([params], lr=1e-2)

    dual_optimizer = torch_coop.optim.partial(torch_coop.optim.ExtraSGD, lr=1e-2)

    construct_closure = functools.partial(closure_2d.construct_closure, use_ineq=True)

    cmp = torch_coop.ConstrainedMinimizationProblem(is_constrained=True)
    formulation = torch_coop.LagrangianFormulation(cmp)

    coop = torch_coop.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optimizer,
        dual_optimizer=dual_optimizer,
        dual_restarts=False,
    )

    # Helper function to instantiate tensors in correct device
    mktensor = functools.partial(torch.tensor, device=device)

    coop.zero_grad()
    closure = construct_closure(params)
    lagrangian = coop.composite_objective(closure)

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
    coop.custom_backward(lagrangian)
    assert torch.allclose(params.grad, mktensor([0.0, -4.0]))
    assert torch.allclose(formulation.state()[0].grad, cmp.state.ineq_defect)

    # Check updated primal and dual variable values
    coop.step(construct_closure, closure_args=params)
    assert torch.allclose(params, mktensor([2.0e-4, -0.9614]))
    assert torch.allclose(formulation.state()[0], mktensor([0.0196, 0.0]))
