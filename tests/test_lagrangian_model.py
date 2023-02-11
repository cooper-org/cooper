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


@pytest.fixture
def get_dummy_cmp():
    return DummyCMP()


@pytest.fixture
def get_ineq_multiplier_model():
    return DummyMultiplierModel()


@pytest.fixture
def get_eq_multiplier_model():
    return DummyMultiplierModel()


def return_fixture_values(cmp, ineq_multiplier_model, eq_multiplier_model, request):
    cmp = request.getfixturevalue(cmp)
    if ineq_multiplier_model is not None:
        ineq_multiplier_model = request.getfixturevalue(ineq_multiplier_model)
    if eq_multiplier_model is not None:
        eq_multiplier_model = request.getfixturevalue(eq_multiplier_model)
    return cmp, ineq_multiplier_model, eq_multiplier_model


@pytest.mark.parametrize("cmp", ["get_dummy_cmp"])
@pytest.mark.parametrize("ineq_multiplier_model", [None, "get_ineq_multiplier_model"])
@pytest.mark.parametrize("eq_multiplier_model", [None, "get_eq_multiplier_model"])
def test_is_state_created(cmp, ineq_multiplier_model, eq_multiplier_model, request):

    cmp, ineq_multiplier_model, eq_multiplier_model = return_fixture_values(
        cmp, ineq_multiplier_model, eq_multiplier_model, request
    )

    if ineq_multiplier_model is None and eq_multiplier_model is None:
        with pytest.raises(ValueError):
            cooper.formulation.LagrangianModelFormulation(
                cmp,
                ineq_multiplier_model=ineq_multiplier_model,
                eq_multiplier_model=eq_multiplier_model,
            )
    else:
        lmf = cooper.formulation.LagrangianModelFormulation(
            cmp,
            ineq_multiplier_model=ineq_multiplier_model,
            eq_multiplier_model=eq_multiplier_model,
        )
        assert lmf.is_state_created


@pytest.mark.parametrize(
    "cmp, ineq_multiplier_model, eq_multiplier_model",
    [
        ("get_dummy_cmp", "get_ineq_multiplier_model", None),
        ("get_dummy_cmp", None, "get_eq_multiplier_model"),
        ("get_dummy_cmp", "get_ineq_multiplier_model", "get_eq_multiplier_model"),
    ],
)
def test_state(cmp, ineq_multiplier_model, eq_multiplier_model, request):

    cmp, ineq_multiplier_model, eq_multiplier_model = return_fixture_values(
        cmp, ineq_multiplier_model, eq_multiplier_model, request
    )

    lmf = cooper.formulation.LagrangianModelFormulation(
        cmp,
        ineq_multiplier_model=ineq_multiplier_model,
        eq_multiplier_model=eq_multiplier_model,
    )

    ineq_state, eq_state = lmf.state()

    if ineq_multiplier_model is None:
        assert ineq_state is None
    else:
        assert ineq_state is not None

    if eq_multiplier_model is None:
        assert eq_state is None
    else:
        assert eq_state is not None


@pytest.mark.parametrize(
    "cmp, ineq_multiplier_model, eq_multiplier_model",
    [
        ("get_dummy_cmp", "get_ineq_multiplier_model", None),
        ("get_dummy_cmp", None, "get_eq_multiplier_model"),
        ("get_dummy_cmp", "get_ineq_multiplier_model", "get_eq_multiplier_model"),
    ],
)
def test_flip_dual_gradients(cmp, ineq_multiplier_model, eq_multiplier_model, request):

    cmp, ineq_multiplier_model, eq_multiplier_model = return_fixture_values(
        cmp, ineq_multiplier_model, eq_multiplier_model, request
    )

    lmf = cooper.formulation.LagrangianModelFormulation(
        cmp,
        ineq_multiplier_model=ineq_multiplier_model,
        eq_multiplier_model=eq_multiplier_model,
    )

    for constraint_type in ["eq", "ineq"]:
        mult_name = constraint_type + "_multiplier_model"
        multiplier_model = getattr(lmf, mult_name)
        if multiplier_model is not None:
            for param in multiplier_model.parameters():
                param.requires_grad = True
                param.grad = torch.ones_like(param)

    lmf.flip_dual_gradients()

    for constraint_type in ["eq", "ineq"]:
        mult_name = constraint_type + "_multiplier_model"
        multiplier_model = getattr(lmf, mult_name)
        if multiplier_model is not None:
            for param in multiplier_model.parameters():
                assert torch.all(param.grad == -1)


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
