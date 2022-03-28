#!/usr/bin/env python

"""Tests for Constrained Optimizer class."""

import functools

import pytest
import testing_utils
import torch

# Import basic closure example from helpers
import toy_2d_problem

import cooper


@pytest.mark.parametrize("aim_device", ["cpu", "cuda"])
def test_toy_problem(aim_device):
    """
    Simple test on a bi-variate quadratic programming problem
        min x**2 + 2*y**2
        st.
            x + y >= 1
            x**2 + y <= 1.

    Verified solution from WolframAlpha (x=2/3, y=1/3)
    Link to WolframAlpha query: https://tinyurl.com/ye8dw6t3
    """

    device, skip = testing_utils.get_device_skip(aim_device, torch.cuda.is_available())

    if skip.do_skip:
        pytest.skip(skip.skip_reason)

    params = torch.nn.Parameter(torch.tensor([0.0, -1.0], device=device))
    primal_optimizer = cooper.optim.SGD([params], lr=5e-2, momentum=0.0)
    dual_optimizer = cooper.optim.partial_optimizer(cooper.optim.SGD, lr=1e-2)

    cmp = toy_2d_problem.Toy2dCMP(use_ineq=True, use_proxy_ineq=True)
    formulation = cooper.LagrangianFormulation(cmp)

    coop = cooper.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optimizer,
        dual_optimizer=dual_optimizer,
        dual_restarts=False,
        alternating=False,
    )

    # Helper function to instantiate tensors in correct device
    mktensor = functools.partial(torch.tensor, device=device)

    # ----------------------- First iteration -----------------------
    coop.zero_grad()
    lagrangian = formulation.composite_objective(cmp.closure, params)

    # Check loss, proxy and non-proxy defects after forward pass
    assert torch.allclose(lagrangian, mktensor(2.0))
    assert torch.allclose(cmp.state.loss, mktensor(2.0))
    assert torch.allclose(cmp.state.ineq_defect, mktensor([2.0, -2.0]))
    assert torch.allclose(cmp.state.proxy_ineq_defect, mktensor([2.0, -1.9]))
    assert cmp.state.eq_defect is None
    assert cmp.state.proxy_eq_defect is None

    # Multiplier initialization
    assert torch.allclose(formulation.state()[0], mktensor([0.0, 0.0]))
    assert formulation.state()[1] is None

    # Check primal and dual gradients after backward. Dual gradient must match
    # ineq_defect
    formulation.custom_backward(lagrangian)
    assert torch.allclose(params.grad, mktensor([0.0, -4.0]))
    assert torch.allclose(formulation.state()[0].grad, cmp.state.ineq_defect)

    # Check updated primal and dual variable values
    coop.step()
    assert torch.allclose(params, mktensor([0.0, -0.8]))
    assert torch.allclose(formulation.state()[0], mktensor([0.02, 0.0]))

    # ----------------------- Second iteration -----------------------
    coop.zero_grad()
    lagrangian = formulation.composite_objective(cmp.closure, params)

    # Check loss, proxy and non-proxy defects after forward pass
    assert torch.allclose(lagrangian, mktensor(1.316))
    assert torch.allclose(cmp.state.loss, mktensor(1.28))
    assert torch.allclose(cmp.state.ineq_defect, mktensor([1.8, -1.8]))
    assert torch.allclose(cmp.state.proxy_ineq_defect, mktensor([1.8, -1.72]))

    # Check primal and dual gradients after backward. Dual gradient must match
    # ineq_defect
    formulation.custom_backward(lagrangian)
    assert torch.allclose(params.grad, mktensor([-0.018, -3.22]))
    assert torch.allclose(formulation.state()[0].grad, cmp.state.ineq_defect)

    # Check updated primal and dual variable values
    coop.step()
    assert torch.allclose(params, mktensor([9e-4, -0.639]))
    assert torch.allclose(formulation.state()[0], mktensor([0.038, 0.0]))

    if device == "cuda":
        assert cmp.state.loss.is_cuda
        assert cmp.state.ineq_defect.is_cuda
