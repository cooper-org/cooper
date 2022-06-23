#!/usr/bin/env python

"""Tests for Extrapolation optimizers."""

import functools
import pdb

import pytest
import testing_utils
import torch

# Import basic closure example from helpers
import toy_2d_problem

import cooper


def prepare_problem(aim_device, alternating):
    device, skip = testing_utils.get_device_skip(aim_device, torch.cuda.is_available())

    if skip.do_skip:
        pytest.skip(skip.skip_reason)

    params = torch.nn.Parameter(torch.tensor([0.0, -1.0], device=device))

    try:
        optimizer_class = getattr(cooper.optim, "SGD")
    except:
        optimizer_class = getattr(torch.optim, "SGD")
    primal_optimizer = optimizer_class([params], lr=1e-2)

    dual_optimizer = cooper.optim.partial_optimizer(torch.optim.SGD, lr=1e-2)

    cmp = toy_2d_problem.Toy2dCMP(use_ineq=True)
    formulation = cooper.LagrangianFormulation(cmp)

    coop = cooper.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optimizer,
        dual_optimizer=dual_optimizer,
        dual_restarts=False,
        alternating=alternating,
    )

    # Helper function to instantiate tensors in correct device
    mktensor = functools.partial(torch.tensor, device=device)

    return params, cmp, coop, formulation, mktensor


@pytest.mark.parametrize("aim_device", ["cpu", "cuda"])
@pytest.mark.parametrize("alternating", [True, False])
def test_manual_alternating(aim_device, alternating):
    """
    Test first step of alternating vs non-alternating GDA updates on toy 2D problem.
    """

    params, cmp, coop, formulation, mktensor = prepare_problem(aim_device, alternating)

    coop.zero_grad()
    lagrangian = formulation.composite_objective(cmp.closure, params)

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
    formulation.custom_backward(lagrangian)
    assert torch.allclose(params.grad, mktensor([0.0, -4.0]))
    assert torch.allclose(formulation.state()[0].grad, cmp.state.ineq_defect)

    # # Check updated primal and dual variable values
    coop.step(cmp.closure, params)
    assert torch.allclose(params, mktensor([0.0, -0.96]))

    if alternating:
        # Loss and violation measurements taken the initialization point [0, -0.96]
        assert torch.allclose(cmp.state.loss, mktensor([1.8432]))
        # The constraint defects are [1.96, -1.96]
        assert torch.allclose(formulation.state()[0], mktensor([1.96 * 1e-2, 0.0]))
    else:
        # Loss and violation measurements taken the initialization point [0, -1]
        assert torch.allclose(cmp.state.loss, mktensor([2.0]))
        # In this case the defects are [2., -2.]
        assert torch.allclose(formulation.state()[0], mktensor([2.0 * 1e-2, 0.0]))


@pytest.mark.parametrize("aim_device", ["cpu", "cuda"])
@pytest.mark.parametrize("alternating", [True])
def test_convergence_alternating(aim_device, alternating):
    """
    Test convergence of alternating GDA updates on toy 2D problem.
    """

    params, cmp, coop, formulation, mktensor = prepare_problem(aim_device, alternating)

    for step_id in range(1500):
        coop.zero_grad()

        # When using the unconstrained formulation, lagrangian = loss
        lagrangian = formulation.composite_objective(cmp.closure, params)
        formulation.custom_backward(lagrangian)

        # Need to pass closure to step function to perform alternating updates
        coop.step(cmp.closure, params)

    assert torch.allclose(params[0], torch.tensor(2.0 / 3.0))
    assert torch.allclose(params[1], torch.tensor(1.0 / 3.0))
