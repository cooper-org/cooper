#!/usr/bin/env python

"""Tests for Constrained Optimizer class."""

import functools

import torch
import torch_coop

import pytest
import testing_utils


@pytest.mark.parametrize("aim_device", ["cpu"])
def test_toy_problem(aim_device):
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

    def construct_closure(params):
        param_x, param_y = params

        def closure_fn():
            # Define toy closure function

            loss = param_x ** 2 + 2 * param_y ** 2
            eq_defect = None  # No equality constraints

            # Two inequality constraints
            ineq_defect = torch.stack(
                [
                    -param_x - param_y + 1.0,  # x + y \ge 1
                    param_x ** 2 + param_y - 1.0,  # x**2 + y \le 1.0
                ]
            )

            closure_state = torch_coop.CMPState(
                loss=loss, ineq_defect=ineq_defect, eq_defect=eq_defect
            )

            return closure_state

        return closure_fn

    params = torch.nn.Parameter(torch.tensor([0.0, -1.0], device=device))
    primal_optimizer = torch_coop.optim.SGD([params], lr=1e-2, momentum=0.3)
    dual_optimizer = functools.partial(torch_coop.optim.SGD, lr=1e-2)

    cmp = torch_coop.ConstrainedMinimizationProblem(is_constrained=True)
    formulation = torch_coop._LagrangianFormulation(cmp)

    coop = torch_coop.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optimizer,
        dual_optimizer=dual_optimizer,
        dual_restarts=True,
    )

    for step_id in range(1500):
        coop.zero_grad()

        closure = construct_closure(params)
        lagrangian = coop.composite_objective(closure)

        coop.custom_backward(lagrangian)

        coop.step(closure)

    if device == "cuda":
        assert cmp.loss.is_cuda
        assert cmp.eq_defect is None or cmp.eq_defect.is_cuda
        assert cmp.ineq_defect is None or cmp.ineq_defect.is_cuda

    assert torch.allclose(params[0], torch.tensor(2.0 / 3.0))
    assert torch.allclose(params[1], torch.tensor(1.0 / 3.0))
