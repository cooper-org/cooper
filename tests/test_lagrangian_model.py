#!/usr/bin/env python

"""Tests for Lagrangian Formulation class."""

import cooper
import pytest
import torch
import random
import numpy as np
from copy import deepcopy

from .helpers import lagrangian_model_test_utils

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

    cmp = DummyCMP()

    mm = lagrangian_model_test_utils.ToyMultiplierModel(10, 10)

    lmf = cooper.formulation.LagrangianModelFormulation(
        cmp, ineq_multiplier_model=mm, eq_multiplier_model=mm
    )

    # Test is_state_created
    assert lmf.is_state_created

    # test lf state
    eq_featurs = torch.randn(100, 10)
    ineq_featurs = torch.randn(100, 10)
    ineq_state, eq_state = lmf.state(ineq_featurs, eq_featurs)

    assert ineq_state is not None
    assert eq_state is not None

    # TODO: evaluate if it is necessary to test dual_parameters and flip_dual_gradients


@pytest.mark.parametrize("aim_device", ["cpu", "cuda"])
def test_convergence_lagrangian_model(aim_device):

    test_problem_data = lagrangian_model_test_utils.build_test_problem(
        aim_device=aim_device,
        primal_init=[0.0, -1.0],
        primal_optim_cls=torch.optim.SGD,
        dual_optim_cls=torch.optim.SGD,
        do_constraint_sampling=False,
        use_proxy_ineq=False,
        primal_optim_kwargs={"lr": 1.5e-1},
        dual_optim_kwargs={"lr": 1.5e-2},
        formulation_cls=cooper.formulation.LagrangianModelFormulation,
    )

    params, cmp, coop, formulation, device, mktensor = test_problem_data.as_tuple()

    coop.instantiate_dual_optimizer_and_scheduler()

    mults = []
    mm_params = [deepcopy(list(formulation.ineq_multiplier_model.parameters()))]
    mm_grads = []
    for step_id in range(200):
        coop.zero_grad()

        lagrangian = formulation.compute_lagrangian(
            closure=cmp.closure,
            params=params,
        )
        formulation.backward(lagrangian)
        if step_id % 5 == 0:
            mults.append(formulation.ineq_multipliers)
            mm_params.append(deepcopy(list(formulation.ineq_multiplier_model.parameters())))
            mm_grads.append(deepcopy(list(formulation.ineq_multiplier_model.grad)))
        coop.step()

    if device == "cuda":
        assert cmp.state.loss.is_cuda
        assert cmp.state.eq_defect is None or cmp.state.eq_defect.is_cuda
        assert cmp.state.ineq_defect is None or cmp.state.ineq_defect.is_cuda

    assert torch.allclose(params[0], mktensor(2.0 / 3.0), atol=1e-3)
    assert torch.allclose(params[1], mktensor(1.0 / 3.0), atol=1e-3)
