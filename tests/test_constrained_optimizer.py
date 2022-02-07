#!/usr/bin/env python

"""Tests for Multiplier class."""

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
    import functools

    dual_optimizer = functools.partial(torch_coop.optim.SGD, lr=1e-2)
    coop = torch_coop.ConstrainedOptimizer(
        primal_optimizer=primal_optimizer, dual_optimizer=dual_optimizer
    )
    assert coop.is_constrained
