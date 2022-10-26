#!/usr/bin/env python

"""Tests for Constrained Optimizer class. This test already verifies that the
code behaves as expected for an unconstrained setting."""

import pytest
import torch

import cooper


@pytest.fixture()
def params():
    return torch.nn.Parameter(torch.tensor([0.0, -1.0]))


@pytest.fixture()
def formulation():
    return cooper.LagrangianFormulation()


@pytest.fixture()
def constrained_optimizer(params, formulation):
    primal_optim = torch.optim.SGD([params], lr=1e-2, momentum=0.3)
    dual_optim = cooper.optim.partial_optimizer(torch.optim.SGD, lr=1e-2)

    return cooper.SimultaneousConstrainedOptimizer(
        formulation, primal_optim, dual_optim, dual_restarts=True
    )


def loss_fn(params):
    param_x, param_y = params

    return param_x**2 + 2 * param_y**2


def defect_fn(params):

    param_x, param_y = params

    # Two inequality constraints
    defect = torch.stack(
        [
            -param_x - param_y + 1.0,  # x + y \ge 1
            param_x**2 + param_y - 1.0,  # x**2 + y \le 1.0
        ]
    )

    return defect


def test_simplest_pipeline(params, formulation, constrained_optimizer):

    for step_id in range(1500):
        constrained_optimizer.zero_grad()

        loss = loss_fn(params)
        defect = defect_fn(params)

        # Create a CMPState object to hold the loss and defect values
        cmp_state = cooper.CMPState(loss=loss, ineq_defect=defect)

        lagrangian = formulation.composite_objective(pre_computed_state=cmp_state)
        formulation.custom_backward(lagrangian)

        constrained_optimizer.step()

    assert torch.allclose(params[0], torch.tensor(2.0 / 3.0))
    assert torch.allclose(params[1], torch.tensor(1.0 / 3.0))
