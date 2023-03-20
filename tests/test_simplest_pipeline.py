#!/usr/bin/env python

"""Tests for Constrained Optimizer class. This test already verifies that the
code behaves as expected for an unconstrained setting."""

from typing import List

import pytest
import torch

import cooper


def evaluate_loss(params):
    param_x, param_y = params
    return param_x**2 + 2 * param_y**2


def evaluate_constraints(params) -> List[cooper.ConstraintState]:
    param_x, param_y = params
    cg0_state = cooper.ConstraintState(violation=-param_x - param_y + 1.0)  # x + y \ge 1
    cg1_state = cooper.ConstraintState(violation=param_x**2 + param_y - 1.0)  # x**2 + y \le 1.0
    return [cg0_state, cg1_state]


def test_simplest_pipeline(Toy2dCMP_params_init, device):
    """Test correct behavior of simultaneous updates on a 2-dimensional constrained
    problem without requiring the user to implement a CMP class explicitly. The only
    required methods are a function to evaluate the loss, and a function to evaluate
    the constraints.
    """

    params = torch.nn.Parameter(Toy2dCMP_params_init)
    primal_optimizer = torch.optim.SGD([params], lr=1e-2, momentum=0.3)

    cg0 = cooper.ConstraintGroup(constraint_type="ineq", formulation_type="lagrangian", shape=1, device=device)
    cg1 = cooper.ConstraintGroup(constraint_type="ineq", formulation_type="lagrangian", shape=1, device=device)
    constraint_groups = [cg0, cg1]

    dual_params = [{"params": constraint.multiplier.parameters()} for constraint in constraint_groups]
    dual_optimizer = torch.optim.SGD(dual_params, lr=1e-2)

    constrained_optimizer = cooper.optim.SimultaneousConstrainedOptimizer(
        primal_optimizers=primal_optimizer, dual_optimizers=dual_optimizer, constraint_groups=constraint_groups
    )

    for step_id in range(1500):
        constrained_optimizer.zero_grad()

        loss = evaluate_loss(params)

        cg0_state, cg1_state = evaluate_constraints(params)
        observed_constraints = [(cg0, cg0_state), (cg1, cg1_state)]

        # # Alternatively, one could assign the constraint states directly to the
        # # constraint groups and collect only the constraint groups when gathering the
        # # observed constraints.
        # cg0.state, cg1.state = evaluate_constraints(params)
        # observed_constraints = [cg0, cg1]

        cmp_state = cooper.CMPState(loss=loss, observed_constraints=observed_constraints)
        lagrangian, multipliers = cmp_state.populate_lagrangian(return_multipliers=True)
        cmp_state.backward()
        constrained_optimizer.step()

    assert torch.allclose(params[0], torch.tensor(2.0 / 3.0))
    assert torch.allclose(params[1], torch.tensor(1.0 / 3.0))
