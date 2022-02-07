#!/usr/bin/env python

"""Tests for Constrained Optimizer class."""

import functools

import torch
import torch_coop


def test_optimizer_init():

    # Dummy wrapper for unconstrained optimization.
    # If only the primal optimizer is provided, coop behaves like a regular optimizaer
    primal_params = torch.nn.Parameter(torch.randn(10, 1))
    primal_optimizer = torch_coop.optim.SGD([primal_params], lr=1e-2)
    coop = torch_coop.ConstrainedOptimizer(primal_optimizer=primal_optimizer)
    assert not coop.is_constrained

    # Create actual constrained optimizer

    dual_optimizer = functools.partial(torch_coop.optim.SGD, lr=1e-2)
    coop = torch_coop.ConstrainedOptimizer(
        primal_optimizer=primal_optimizer, dual_optimizer=dual_optimizer
    )
    assert coop.is_constrained


def test_toy_problem():
    """
    Simple test on a bi-variate quadratic programming problem
        min x**2 + 2*y**2
        st.
            x + y >= 1
            x**2 + y <= 1

    Verified solution from WolframAlpha (x=2/3, y=1/3)
    Link to WolframAlpha query: https://tinyurl.com/ye8dw6t3
    """

    def construct_closure(params):
        param_x, param_y = params

        def closure():
            # Define toy closure function
            closure_dict = {
                "loss": param_x ** 2 + 2 * param_y ** 2,
                "eq_defect": [],
                "ineq_defect": [],
            }

            # x + y \ge 1
            closure_dict["ineq_defect"].append(-param_x - param_y + 1.0)
            # x**2 + y \le 0.5
            closure_dict["ineq_defect"].append(param_x ** 2 + param_y - 1.0)

            return closure_dict

        return closure

    params = torch.nn.Parameter(torch.tensor([0.0, -1.0]))
    primal_optimizer = torch_coop.optim.SGD([params], lr=1e-2, momentum=0.3)
    dual_optimizer = functools.partial(torch_coop.optim.SGD, lr=1e-2)

    coop = torch_coop.ConstrainedOptimizer(
        primal_optimizer=primal_optimizer, dual_optimizer=dual_optimizer
    )

    for step_id in range(1500):
        _closure_dict = coop.step(construct_closure(params))

    assert torch.allclose(params[0], torch.tensor(2.0 / 3.0))
    assert torch.allclose(params[1], torch.tensor(1.0 / 3.0))
