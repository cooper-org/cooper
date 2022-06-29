#!/usr/bin/env python

"""Tests for Constrained Optimizer class."""

import pdb

import cooper_test_utils
import pytest
import torch

import cooper


@pytest.mark.parametrize("aim_device", ["cpu", "cuda"])
def test_manual_alternating_proxy(aim_device):
    """ """

    test_problem_data = cooper_test_utils.build_test_problem(
        aim_device=aim_device,
        primal_optim_cls=torch.optim.SGD,
        primal_init=[0.0, -1.0],
        dual_optim_cls=torch.optim.SGD,
        use_ineq=True,
        use_proxy_ineq=True,
        dual_restarts=False,
        alternating=True,
        primal_optim_kwargs={"lr": 5e-2, "momentum": 0.0},
        dual_optim_kwargs={"lr": 1e-2},
    )

    params, cmp, coop, formulation, device, mktensor = test_problem_data.as_tuple()

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

    # Must pass closrue again to compute constraints for alternating update
    coop.step(cmp.closure, params)

    assert torch.allclose(params, mktensor([0.0, -0.8]))
    # Loss and constraints are evaluated at updated primal variables
    assert torch.allclose(cmp.state.loss, mktensor([2 * (-0.8) ** 2]))
    # Constraint defects [1.8, -1.8] --> this is used to update multipliers
    # "Proxy defects" [1.8, -1.72] --> used to compute primal gradient
    assert torch.allclose(formulation.state()[0], mktensor([1.8 * 1e-2, 0.0]))

    # ----------------------- Second iteration -----------------------
    coop.zero_grad()
    lagrangian = formulation.composite_objective(cmp.closure, params)

    # Check loss, proxy and non-proxy defects after forward pass
    assert torch.allclose(lagrangian, mktensor(1.3124))
    assert torch.allclose(cmp.state.loss, mktensor(1.28))
    assert torch.allclose(cmp.state.ineq_defect, mktensor([1.8, -1.8]))
    assert torch.allclose(cmp.state.proxy_ineq_defect, mktensor([1.8, -1.72]))

    # Check primal and dual gradients after backward. Dual gradient must match
    # ineq_defect
    formulation.custom_backward(lagrangian)
    assert torch.allclose(params.grad, mktensor([-0.0162, -3.218]))
    assert torch.allclose(formulation.state()[0].grad, cmp.state.ineq_defect)

    # Must pass closrue again to compute constraints for alternating update
    coop.step(cmp.closure, params)

    assert torch.allclose(params, mktensor([8.1e-4, -0.6391]))
    # Constraint violation at this point [1.6383, -1.63909936]
    assert torch.allclose(formulation.state()[0], mktensor([0.034383, 0.0]))

    if device == "cuda":
        assert cmp.state.loss.is_cuda
        assert cmp.state.ineq_defect.is_cuda
