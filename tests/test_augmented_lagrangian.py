#!/usr/bin/env python

"""Tests for Augmented Lagrangian Formulation class."""

import functools
import pdb

import pytest
import testing_utils
import torch
import toy_2d_problem

import cooper


def test_augmented_lagrangian_formulation():
    class DummyCMP(cooper.ConstrainedMinimizationProblem):
        def __init__(self):
            super().__init__(is_constrained=True)

        def closure(self):
            pass

    cmp = DummyCMP()

    formulation = cooper.AugmentedLagrangianFormulation(cmp, aug_lag_coefficient=1.0)
    cmp.state = cooper.CMPState(eq_defect=torch.tensor([1.0]))
    formulation.create_state(cmp.state)

    assert (formulation.ineq_multipliers is None) and (
        formulation.eq_multipliers is not None
    )

    formulation = cooper.AugmentedLagrangianFormulation(cmp, aug_lag_coefficient=1.0)
    cmp.state = cooper.CMPState(
        eq_defect=torch.tensor([1.0]), ineq_defect=torch.tensor([1.0, 1.2])
    )
    formulation.create_state(cmp.state)
    assert (formulation.ineq_multipliers is not None) and (
        formulation.eq_multipliers is not None
    )


@pytest.mark.parametrize("aim_device", ["cpu", "cuda"])
def test_convergence_augmented_lagrangian(aim_device):
    """ """

    device, skip = testing_utils.get_device_skip(aim_device, torch.cuda.is_available())

    # Helper function to instantiate tensors in correct device
    mktensor = functools.partial(torch.tensor, device=device)

    if skip.do_skip:
        pytest.skip(skip.skip_reason)

    params = torch.nn.Parameter(torch.tensor([0.0, -1.0], device=device))
    primal_optimizer = torch.optim.SGD([params], lr=1e-2)

    cmp = toy_2d_problem.Toy2dCMP(use_ineq=True)

    dual_optimizer = cooper.optim.partial_optimizer(torch.optim.SGD, lr=1.0)
    formulation = cooper.AugmentedLagrangianFormulation(cmp, aug_lag_coefficient=1.0)
    # formulation = cooper.LagrangianFormulation(cmp)

    coop = cooper.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optimizer,
        dual_optimizer=dual_optimizer,
        dual_restarts=False,
        alternating=True,
    )

    for step_id in range(1500):
        coop.zero_grad()

        # When using the unconstrained formulation, lagrangian = loss
        lagrangian = formulation.composite_objective(cmp.closure, params)
        formulation.custom_backward(lagrangian)

        coop.step(cmp.closure, params)

    if device == "cuda":
        assert cmp.state.loss.is_cuda
        assert cmp.state.eq_defect is None or cmp.state.eq_defect.is_cuda
        assert cmp.state.ineq_defect is None or cmp.state.ineq_defect.is_cuda

    assert torch.allclose(params[0], mktensor(2.0 / 3.0))
    assert torch.allclose(params[1], mktensor(1.0 / 3.0))


@pytest.mark.parametrize("aim_device", ["cpu", "cuda"])
def test_manual_augmented_lagrangian(aim_device):
    """ """

    device, skip = testing_utils.get_device_skip(aim_device, torch.cuda.is_available())

    if skip.do_skip:
        pytest.skip(skip.skip_reason)

    # Helper function to instantiate tensors in correct device
    mktensor = functools.partial(torch.tensor, device=device)

    params = torch.nn.Parameter(torch.tensor([0.0, -1.0], device=device))
    primal_optimizer = torch.optim.SGD([params], lr=1e-2, momentum=0.0)

    cmp = toy_2d_problem.Toy2dCMP(use_ineq=True)

    dual_optimizer = cooper.optim.partial_optimizer(torch.optim.SGD, lr=1.0)
    formulation = cooper.AugmentedLagrangianFormulation(cmp, aug_lag_coefficient=1.0)
    # formulation = cooper.LagrangianFormulation(cmp)

    coop = cooper.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optimizer,
        dual_optimizer=dual_optimizer,
        dual_restarts=False,
        alternating=True,
    )

    # ---------- First iteration ----------

    coop.zero_grad()

    # When using the unconstrained formulation, lagrangian = loss
    lagrangian = formulation.composite_objective(cmp.closure, params)

    assert torch.allclose(cmp.state.loss, mktensor(2.0))
    assert torch.allclose(lagrangian, mktensor(4.0))

    formulation.custom_backward(lagrangian)

    assert torch.allclose(params.grad, mktensor([-2.0, -6.0]))

    coop.step(cmp.closure, params)
    assert torch.allclose(params, mktensor([0.02, -0.94]))

    # Check inequality multipliers
    assert torch.allclose(formulation.state()[0], mktensor([1.92, 0.0]))

    # ---------- Second iteration ----------

    coop.zero_grad()

    # When using the unconstrained formulation, lagrangian = loss
    lagrangian = formulation.composite_objective(cmp.closure, params)

    assert torch.allclose(cmp.state.loss, mktensor(1.7676))
    assert torch.allclose(lagrangian, mktensor(7.2972))

    formulation.custom_backward(lagrangian)

    assert torch.allclose(params.grad, mktensor([-3.8, -7.6]))

    coop.step(cmp.closure, params)
    assert torch.allclose(params, mktensor([0.058, -0.864]))

    # Check inequality multipliers
    # Multiplier gradient signed is flipped inside step
    assert torch.allclose(-formulation.state()[0].grad, mktensor([1.8060, -1.860636]))
    assert torch.allclose(formulation.state()[0], mktensor([3.726, 0.0]))
