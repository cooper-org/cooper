#!/usr/bin/env python

"""Tests for Lagrangian Formulation class."""

import random

import cooper
import numpy as np
import pytest
import torch

from .helpers import cooper_test_utils

random.seed(121212)
np.random.seed(121212)
torch.manual_seed(121212)
torch.cuda.manual_seed(121211)


def test_lagrangian_formulation():
    class DummyCMP(cooper.ConstrainedMinimizationProblem):
        def __init__(self):
            super().__init__()

        def closure(self):
            pass

    cmp = DummyCMP()

    lf = cooper.LagrangianFormulation(cmp)
    cmp.state = cooper.CMPState(eq_defect=torch.tensor([1.0]))
    lf.create_state(cmp.state)
    assert (lf.ineq_multipliers is None) and (lf.eq_multipliers is not None)

    lf = cooper.LagrangianFormulation(cmp)
    cmp.state = cooper.CMPState(
        eq_defect=torch.tensor([1.0]), ineq_defect=torch.tensor([1.0, 1.2])
    )
    lf.create_state(cmp.state)
    assert (lf.ineq_multipliers is not None) and (lf.eq_multipliers is not None)


@pytest.mark.parametrize("aim_device", ["cpu", "cuda"])
def test_convergence_small_toy_problem(aim_device):

    test_problem_data = cooper_test_utils.build_test_problem(
        aim_device=aim_device,
        primal_optim_cls=torch.optim.SGD,
        primal_init=[0.0, -1.0],
        dual_optim_cls=torch.optim.SGD,
        use_ineq=True,
        use_proxy_ineq=False,
        use_mult_model=False,
        dual_restarts=False,
        alternating=True,
        primal_optim_kwargs={"lr": 1e-2},
        dual_optim_kwargs={"lr": 1.0},
        dual_scheduler=None,
        formulation_cls=cooper.formulation.LagrangianFormulation,
    )

    params, cmp, coop, formulation, device, mktensor = test_problem_data.as_tuple()

    for step_id in range(400):
        coop.zero_grad()
        lagrangian = formulation.compute_lagrangian(
            closure=cmp.closure,
            params=params,
        )
        formulation.backward(lagrangian)
        coop.step(cmp.closure, params)

    if device == "cuda":
        assert cmp.state.loss.is_cuda
        assert cmp.state.eq_defect is None or cmp.state.eq_defect.is_cuda
        assert cmp.state.ineq_defect is None or cmp.state.ineq_defect.is_cuda

    assert torch.allclose(params[0], mktensor(2.0 / 3.0), atol=1e-3)
    assert torch.allclose(params[1], mktensor(1.0 / 3.0), atol=1e-3)
