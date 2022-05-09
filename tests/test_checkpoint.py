#!/usr/bin/env python

"""Tests for checkpointing. This test already verifies that checkpointing works
for the unconstrained setting."""

import os
import tempfile

import pytest
import pytorch_testing_utils as ptu
import testing_utils
import torch

# Import basic closure example from helpers
import toy_2d_problem

import cooper


def train_for_n_steps(coop, cmp, params, n_step=100):

    for _ in range(n_step):
        coop.zero_grad()

        # When using the unconstrained formulation, lagrangian = loss
        lagrangian = coop.formulation.composite_objective(cmp.closure, params)
        coop.formulation.custom_backward(lagrangian)

        coop.step()


class Model(torch.nn.Module):
    def __init__(self, params=None):
        super(Model, self).__init__()

        if params is not None:
            self.register_parameter(name="params", param=torch.nn.Parameter(params))
        else:
            self.params = torch.nn.Parameter(torch.tensor([0.0, 0.0]))

    def forward(self):
        return self.params


@pytest.mark.parametrize("aim_device", ["cpu", "cuda"])
@pytest.mark.parametrize("use_ineq", [True, False])
def test_toy_problem(aim_device, use_ineq):
    """
    Simple test on a bi-variate quadratic programming problem
        min x**2 + 2*y**2
        st.
            x + y >= 1
            x**2 + y <= 1

    Verified solution from WolframAlpha (x=2/3, y=1/3)
    Link to WolframAlpha query: https://tinyurl.com/ye8dw6t3
    """

    device, skip = testing_utils.get_device_skip(aim_device, torch.cuda.is_available())

    if skip.do_skip:
        pytest.skip(skip.skip_reason)

    model = Model(torch.tensor([0.0, -1.0], device=device))

    partial_primal_optim = cooper.optim.partial_optimizer(
        torch.optim.SGD, lr=1e-2, momentum=0.3
    )
    primal_optimizer = partial_primal_optim(model.parameters())

    cmp = toy_2d_problem.Toy2dCMP(use_ineq=use_ineq)

    if use_ineq:
        # Constrained case
        partial_dual_optim = cooper.optim.partial_optimizer(torch.optim.SGD, lr=1e-2)
        formulation = cooper.LagrangianFormulation(cmp)
    else:
        # Unconstrained case
        partial_dual_optim = None
        formulation = cooper.UnconstrainedFormulation(cmp)

    coop = cooper.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optimizer,
        dual_optimizer=partial_dual_optim,
        dual_restarts=True,
    )

    # Train for 100 steps
    train_for_n_steps(coop, cmp, model, n_step=10)

    # Generate checkpoints after 100 steps of training
    model_state_dict_100 = model.state_dict()
    form_state_dict_100 = coop.formulation.state_dict()
    coop_state_dict_100 = coop.state_dict()

    with tempfile.TemporaryDirectory() as tmpdirname:
        torch.save(model_state_dict_100, os.path.join(tmpdirname, "model.pth"))
        torch.save(form_state_dict_100, os.path.join(tmpdirname, "formulation.pth"))
        torch.save(coop_state_dict_100, os.path.join(tmpdirname, "coop.pth"))

        del model_state_dict_100
        del form_state_dict_100
        del coop_state_dict_100

        model_state_dict_100 = torch.load(os.path.join(tmpdirname, "model.pth"))
        form_state_dict_100 = torch.load(os.path.join(tmpdirname, "formulation.pth"))
        coop_state_dict_100 = torch.load(os.path.join(tmpdirname, "coop.pth"))

    # pdb.set_trace()

    # Train for another 200 steps
    train_for_n_steps(coop, cmp, model, n_step=100)

    model_state_dict_200 = model.state_dict()
    form_state_dict_200 = coop.formulation.state_dict()
    coop_state_dict_200 = coop.state_dict()

    # Reload from 100-step checkpoint
    loaded_model = Model(None)
    loaded_model.load_state_dict(model_state_dict_100)

    if use_ineq:
        loaded_formulation = cooper.LagrangianFormulation(cmp)
    else:
        loaded_formulation = cooper.UnconstrainedFormulation(cmp)
    loaded_formulation.load_state_dict(form_state_dict_100)

    loaded_coop = cooper.ConstrainedOptimizer.load_from_state_dict(
        const_optim_state=coop_state_dict_100,
        formulation=loaded_formulation,
        primal_optimizer_class=partial_primal_optim,
        primal_parameters=loaded_model.parameters(),
        dual_optimizer_class=partial_dual_optim,
        dual_scheduler_class=None,
    )

    # Train checkpointed model for 100 steps to reach overall 200 steps
    train_for_n_steps(loaded_coop, cmp, loaded_model, n_step=100)

    # Compare 0-200 state_dicts versus the 0-100;100-200 state_dicts
    assert ptu.validate_state_dicts(loaded_model.state_dict(), model_state_dict_200)
    assert ptu.validate_state_dicts(
        loaded_formulation.state_dict(), form_state_dict_200
    )
    # These are ConstrainedOptimizerState objects and not dicts
    assert loaded_coop.state_dict() == coop_state_dict_200
