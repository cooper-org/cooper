#!/usr/bin/env python

"""Tests for Constrained Optimizer class. This test already verifies that the
code behaves as expected for an unconstrained setting."""

import cooper_test_utils
import pytest
import torch


@pytest.mark.parametrize("aim_device", ["cpu", "cuda"])
@pytest.mark.parametrize("use_ineq", [True, False])
@pytest.mark.parametrize("multiple_optimizers", [True, False])
def test_toy_problem(aim_device, use_ineq, multiple_optimizers):
    """
    Verify constrained and unconstrained executions run correctly on a toy 2D
    problem.
    """
    if multiple_optimizers:
        primal_optim_cls = [torch.optim.SGD, torch.optim.Adam]
        primal_optim_kwargs = [{"lr": 1e-2, "momentum": 0.3}, {"lr": 1e-2}]
    else:
        primal_optim_cls = torch.optim.SGD
        primal_optim_kwargs = {"lr": 1e-2, "momentum": 0.3}

    test_problem_data = cooper_test_utils.build_test_problem(
        aim_device=aim_device,
        primal_optim_cls=primal_optim_cls,
        primal_init=[0.0, -1.0],
        dual_optim_cls=torch.optim.SGD,
        use_ineq=use_ineq,
        use_proxy_ineq=False,
        dual_restarts=True,
        alternating=False,
        primal_optim_kwargs=primal_optim_kwargs,
        dual_optim_kwargs={"lr": 1e-2},
    )

    params, cmp, coop, formulation, device, mktensor = test_problem_data.as_tuple()

    for step_id in range(1500):
        coop.zero_grad()

        # When using the unconstrained formulation, lagrangian = loss
        lagrangian = formulation.composite_objective(cmp.closure, params)
        formulation.backward(lagrangian)

        coop.step()

    if device == "cuda":
        assert cmp.state.loss.is_cuda
        assert cmp.state.eq_defect is None or cmp.state.eq_defect.is_cuda
        assert cmp.state.ineq_defect is None or cmp.state.ineq_defect.is_cuda

    if use_ineq:
        assert torch.allclose(params[0], torch.tensor(2.0 / 3.0))
        assert torch.allclose(params[1], torch.tensor(1.0 / 3.0))
    else:
        # This unconstrained quadratic form has minimum at the origin
        assert torch.allclose(params[0], torch.tensor(0.0))
        assert torch.allclose(params[1], torch.tensor(0.0))
