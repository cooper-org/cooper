#!/usr/bin/env python

"""Tests for Lagrangian Model Formulation class."""

import random

import numpy as np
import pytest
import torch

import cooper

from .helpers import cooper_test_utils

random.seed(121212)
np.random.seed(121212)
torch.manual_seed(121212)
torch.cuda.manual_seed(121211)


def test_lagrangian_model():
    class DummyCMP(cooper.ConstrainedMinimizationProblem):
        def __init__(self):
            super().__init__()

        def closure(self):
            pass

    class DummyMultiplierModel(cooper.multipliers.MultiplierModel):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 1)

        def forward(self, constraint_features):
            return self.linear(constraint_features)

    cmp = DummyCMP()
    mm = DummyMultiplierModel()

    # Test is_state_created
    lmf = cooper.formulation.LagrangianModelFormulation(
        cmp, eq_multiplier_model=mm
    )
    assert lmf.is_state_created

    lmf = cooper.formulation.LagrangianModelFormulation(
        cmp, ineq_multiplier_model=mm, eq_multiplier_model=mm
    )
    assert lmf.is_state_created

    # Test state()
    eq_featurs = torch.randn(100, 10)
    ineq_featurs = torch.randn(100, 10)
    ineq_state, eq_state = lmf.state(ineq_featurs, eq_featurs)

    assert ineq_state is not None
    assert eq_state is not None


    # TODO: evaluate if it is necessary to test dual_parameters and flip_dual_gradients


@pytest.mark.parametrize("aim_device", ["cpu", "cuda"])
def test_convergence_small_toy_problem(aim_device):

    test_problem_data = cooper_test_utils.build_test_problem(
        aim_device=aim_device,
        primal_optim_cls=torch.optim.SGD,
        primal_init=[0.0, -1.0],
        dual_optim_cls=torch.optim.SGD,
        use_ineq=True,
        use_proxy_ineq=False,
        use_mult_model=True,
        dual_restarts=False,
        alternating=False,
        primal_optim_kwargs={"lr": 1e-2},
        dual_optim_kwargs={"lr": 1e-2},
        dual_scheduler=None,
        formulation_cls=cooper.formulation.LagrangianModelFormulation,
    )

    params, cmp, coop, formulation, device, mktensor = test_problem_data.as_tuple()

    coop.instantiate_dual_optimizer_and_scheduler()

    for _ in range(400):
        coop.zero_grad()

        lagrangian = formulation.compute_lagrangian(
            closure=cmp.closure,
            params=params,
        )
        formulation.backward(lagrangian)
        coop.step()

    if device == "cuda":
        assert cmp.state.loss.is_cuda
        assert cmp.state.eq_defect is None or cmp.state.eq_defect.is_cuda
        assert cmp.state.ineq_defect is None or cmp.state.ineq_defect.is_cuda

    assert torch.allclose(params[0], mktensor(2.0 / 3.0), atol=1e-2)
    assert torch.allclose(params[1], mktensor(1.0 / 3.0), atol=1e-2)
