#!/usr/bin/env python

"""Tests for Constrained Optimizer class."""

import cooper_test_utils
import pytest
import torch


@pytest.mark.parametrize("aim_device", ["cpu", "cuda"])
def test_manual_proxy_constraints(aim_device):
    """
    Checks correct behavior when using proxy constraints by comparing the
    problem and formulation states over a couple of initial iterations.
    """

    test_problem_data = cooper_test_utils.build_test_problem(
        aim_device=aim_device,
        primal_optim_cls=torch.optim.SGD,
        primal_init=[0.0, -1.0],
        dual_optim_cls=torch.optim.SGD,
        use_ineq=True,
        use_proxy_ineq=True,
        dual_restarts=False,
        alternating=False,
        primal_optim_kwargs={"lr": 5e-2},
        dual_optim_kwargs={"lr": 1e-2},
    )

    params, cmp, coop, formulation, device, mktensor = test_problem_data.as_tuple()

    # ----------------------- First iteration -----------------------
    coop.zero_grad()
    lagrangian = formulation.compute_lagrangian(cmp.closure, params)

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
    formulation.backward(lagrangian)
    assert torch.allclose(params.grad, mktensor([0.0, -4.0]))
    assert torch.allclose(formulation.state()[0].grad, cmp.state.ineq_defect)

    # Check updated primal and dual variable values
    coop.step()
    assert torch.allclose(params, mktensor([0.0, -0.8]))
    assert torch.allclose(formulation.state()[0], mktensor([0.02, 0.0]))

    # ----------------------- Second iteration -----------------------
    coop.zero_grad()
    lagrangian = formulation.compute_lagrangian(cmp.closure, params)

    # Check loss, proxy and non-proxy defects after forward pass
    assert torch.allclose(lagrangian, mktensor(1.316))
    assert torch.allclose(cmp.state.loss, mktensor(1.28))
    assert torch.allclose(cmp.state.ineq_defect, mktensor([1.8, -1.8]))
    assert torch.allclose(cmp.state.proxy_ineq_defect, mktensor([1.8, -1.72]))

    # Check primal and dual gradients after backward. Dual gradient must match
    # ineq_defect
    formulation.backward(lagrangian)
    assert torch.allclose(params.grad, mktensor([-0.018, -3.22]))
    assert torch.allclose(formulation.state()[0].grad, cmp.state.ineq_defect)

    # Check updated primal and dual variable values
    coop.step()
    assert torch.allclose(params, mktensor([9e-4, -0.639]))
    assert torch.allclose(formulation.state()[0], mktensor([0.038, 0.0]))

    if device == "cuda":
        assert cmp.state.loss.is_cuda
        assert cmp.state.ineq_defect.is_cuda
