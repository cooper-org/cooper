#!/usr/bin/env python

"""Tests for Constrained Optimizer class."""

import functools

import torch
import torch_coop

import pytest
import testing_utils

# Import basic closure example from helpers
import closure_2d


@pytest.mark.parametrize("aim_device", ["cpu", "cuda"])
@pytest.mark.parametrize("use_ineq", [True, False])
def test_toy_problem(aim_device, use_ineq):
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

    construct_closure = functools.partial(
        closure_2d.construct_closure, use_ineq=use_ineq
    )

    params = torch.nn.Parameter(torch.tensor([0.0, -1.0], device=device))
    primal_optimizer = torch_coop.optim.SGD([params], lr=1e-2, momentum=0.3)

    if use_ineq:
        dual_optimizer = torch_coop.optim.partial(torch_coop.optim.SGD, lr=1e-2)
    else:
        dual_optimizer = None

    cmp = torch_coop.ConstrainedMinimizationProblem(is_constrained=use_ineq)
    formulation = torch_coop.LagrangianFormulation(cmp)

    coop = torch_coop.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optimizer,
        dual_optimizer=dual_optimizer,
        dual_restarts=True,
    )

    for step_id in range(1500):
        coop.zero_grad()

        closure = construct_closure(params, use_ineq=use_ineq)
        lagrangian = coop.composite_objective(closure)

        coop.custom_backward(lagrangian)

        coop.step(closure)

    if device == "cuda":
        assert cmp.state.loss.is_cuda
        assert cmp.state.eq_defect is None or cmp.state.eq_defect.is_cuda
        assert cmp.state.ineq_defect is None or cmp.state.ineq_defect.is_cuda

    if use_ineq:
        assert torch.allclose(params[0], torch.tensor(2.0 / 3.0))
        assert torch.allclose(params[1], torch.tensor(1.0 / 3.0))
    else:
        # This unconstrained quadratic form has minimum at the origin
        assert torch.allclose(params, torch.tensor(0.0))
