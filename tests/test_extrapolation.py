#!/usr/bin/env python

"""Tests for Extrapolation optimizers."""

import functools

import torch
import torch_coop

import pytest
import testing_utils

# Import basic closure example from helpers
import closure_2d


@pytest.mark.parametrize("aim_device", ["cpu", "cuda"])
@pytest.mark.parametrize("primal_optimizer", ["ExtraSGD", "ExtraAdam"])
def test_extrapolation(aim_device, primal_optimizer):
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
    assert torch.allclose(params[0], torch.tensor(2.0 / 3.0), atol=1e-3)
    assert torch.allclose(params[1], torch.tensor(1.0 / 3.0), atol=1e-3)
