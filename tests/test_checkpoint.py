#!/usr/bin/env python

"""Tests for checkpointing. This test already verifies that checkpointing works
for the unconstrained setting."""

import os
import tempfile

# Import basic closure example from helpers
import cooper_test_utils
import pytest
import torch

import cooper
from cooper.utils import validate_state_dicts


def train_for_n_steps(coop, cmp, params, n_step=100):

    for _ in range(n_step):
        coop.zero_grad()

        # When using the unconstrained formulation, lagrangian = loss
        lagrangian = cooper.compute_lagrangian(formulation, cmp.closure, params)
        cooper.backward(coop.formulation, lagrangian)
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
@pytest.mark.parametrize("multiple_optimizers", [True, False])
def test_checkpoint(aim_device, use_ineq, multiple_optimizers):
    """
    Test that checkpointing and loading works for the constrained and
    unconstrained cases.
    """

    model = Model(torch.tensor([0.0, -1.0]))

    if multiple_optimizers:
        primal_optim_cls = [torch.optim.SGD, torch.optim.Adam]
        primal_optim_kwargs = [{"lr": 1e-2, "momentum": 0.3}, {"lr": 1e-2}]
    else:
        primal_optim_cls = torch.optim.SGD
        primal_optim_kwargs = {"lr": 1e-2, "momentum": 0.3}

    if use_ineq:
        # Constrained case
        partial_dual_optim = cooper.optim.partial_optimizer(torch.optim.SGD, lr=1e-2)
        partial_dual_sch = cooper.optim.partial_scheduler(
            torch.optim.lr_scheduler.StepLR, gamma=0.99, step_size=1
        )
    else:
        # Unconstrained case
        partial_dual_optim = None
        partial_dual_sch = None

    test_problem_data = cooper_test_utils.build_test_problem(
        aim_device=aim_device,
        primal_optim_cls=primal_optim_cls,
        primal_init=None,
        dual_optim_cls=torch.optim.SGD,
        use_ineq=use_ineq,
        use_proxy_ineq=False,
        dual_restarts=True,
        alternating=False,
        primal_optim_kwargs=primal_optim_kwargs,
        dual_optim_kwargs={"lr": 1e-2},
        primal_model=model,
        dual_scheduler=partial_dual_sch,
    )

    params, cmp, coop, formulation, device, mktensor = test_problem_data.as_tuple()

    # Train for 100 steps
    train_for_n_steps(coop, cmp, model, n_step=100)

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

    # Train for another 100 steps
    train_for_n_steps(coop, cmp, model, n_step=100)

    model_state_dict_200 = model.state_dict()
    form_state_dict_200 = coop.formulation.state_dict()
    coop_state_dict_200 = coop.state_dict()

    # Reload from 100-step checkpoint
    loaded_model = Model(None)
    loaded_model.load_state_dict(model_state_dict_100)
    loaded_model.to(device)

    if multiple_optimizers:
        loaded_primal_optimizers = []
        for p, cls, kwargs in zip(params, primal_optim_cls, primal_optim_kwargs):
            loaded_primal_optimizers.append(cls([p], **kwargs))
    else:
        loaded_primal_optimizers = [
            primal_optim_cls(loaded_model.parameters(), **primal_optim_kwargs)
        ]

    if use_ineq:
        loaded_formulation = cooper.LagrangianFormulation(cmp)
    else:
        loaded_formulation = cooper.UnconstrainedFormulation(cmp)
    loaded_formulation.load_state_dict(form_state_dict_100)

    loaded_coop = cooper.optim.load_cooper_optimizer_from_state_dict(
        cooper_optimizer_state=coop_state_dict_100,
        formulation=loaded_formulation,
        primal_optimizers=loaded_primal_optimizers,
        dual_optimizer_class=partial_dual_optim,
        dual_scheduler_class=partial_dual_sch,
    )

    # Train checkpointed model for 100 steps to reach overall 200 steps
    train_for_n_steps(loaded_coop, cmp, loaded_model, n_step=100)

    # Compare 0-200 state_dicts versus the 0-100;100-200 state_dicts
    assert validate_state_dicts(loaded_model.state_dict(), model_state_dict_200)
    assert validate_state_dicts(loaded_formulation.state_dict(), form_state_dict_200)
    # These are ConstrainedOptimizerState objects and not dicts
    assert loaded_coop.state_dict() == coop_state_dict_200
