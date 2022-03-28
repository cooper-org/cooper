#!/usr/bin/env python

"""Tests for Extrapolation optimizers."""

import pytest
import torch

# Import basic closure example from helpers
import toy_2d_problem

import cooper


@pytest.mark.parametrize("scheduler_name", ["ExponentialLR", "ConstantLR"])
@pytest.mark.parametrize("optimizer_name", ["SGD", "ExtraSGD"])
def test_extrapolation(scheduler_name, optimizer_name):
    """
    Simple test on a bi-variate quadratic programming problem
        min x**2 + 2*y**2
        st.
            x + y >= 1
            x**2 + y <= 1
    """

    params = torch.nn.Parameter(torch.tensor([0.0, -1.0]))

    cmp = toy_2d_problem.Toy2dCMP(use_ineq=True)
    formulation = cooper.LagrangianFormulation(cmp)

    optimizer_class = getattr(cooper.optim, optimizer_name)
    primal_optimizer = optimizer_class([params], lr=1e1)
    dual_optimizer = cooper.optim.partial_optimizer(optimizer_class, lr=1e1)

    scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name)
    if scheduler_name == "ExponentialLR":
        scheduler_kwargs = {"gamma": 0.1}
    elif scheduler_name == "ConstantLR":
        scheduler_kwargs = {"factor": 0.5, "total_iters": 4}
    primal_scheduler = scheduler_class(primal_optimizer, **scheduler_kwargs)
    dual_scheduler = cooper.optim.partial_scheduler(scheduler_class, **scheduler_kwargs)

    constrained_optimizer = cooper.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optimizer,
        dual_optimizer=dual_optimizer,
        dual_scheduler=dual_scheduler,
        dual_restarts=False,
    )

    for step_id in range(5):
        constrained_optimizer.zero_grad()
        lagrangian = formulation.composite_objective(cmp.closure, params)
        formulation.custom_backward(lagrangian)

        if hasattr(primal_optimizer, "extrapolation"):
            # Only one dual_scheduler step should be performed even if
            # extrapolation is used
            constrained_optimizer.step(cmp.closure, params)
        else:
            constrained_optimizer.step()

        primal_scheduler.step()

        # Check that the dual learning rate is decreasing correctly
        primal_lr = primal_scheduler.get_last_lr()
        dual_lr = constrained_optimizer.dual_scheduler.get_last_lr()

        if scheduler_name == "ExponentialLR":
            target = torch.tensor(0.1) ** step_id
        elif scheduler_name == "ConstantLR":
            target = torch.tensor(5.0) if step_id < 3 else torch.tensor(10.0)

        assert torch.allclose(torch.tensor(primal_lr), target)
        assert torch.allclose(torch.tensor(dual_lr), target)
