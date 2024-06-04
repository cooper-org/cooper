"""Tests for checkpointing of constrained and unconstrained experiments."""

import os
import tempfile
from typing import Sequence

import pytest
import torch

import cooper
from tests.helpers import cooper_test_utils, testing_utils

PRIMAL_LR = 1e-2
DUAL_LR = 1e-2


class Model(torch.nn.Module):
    def __init__(self, params: Sequence):
        super().__init__()
        self.num_params = len(params)

        for i, param in enumerate(params):
            if not isinstance(param, torch.nn.Parameter):
                param = torch.nn.Parameter(param)
            self.register_parameter(name=f"params_{i}", param=param)

    def forward(self):
        return torch.stack([getattr(self, f"params_{i}") for i in range(self.num_params)])


def construct_cmp(
    multiplier_type,
    num_constraints,
    num_variables,
    device,
):
    A = torch.randn(
        num_constraints, num_variables, device=device, generator=torch.Generator(device=device).manual_seed(0)
    )
    b = torch.randn(num_constraints, device=device, generator=torch.Generator(device=device).manual_seed(0))

    return cooper_test_utils.SquaredNormLinearCMP(
        has_ineq_constraint=True,
        ineq_multiplier_type=multiplier_type,
        ineq_formulation_type=cooper.LagrangianFormulation,
        A=A,
        b=b,
        device=device,
    )


def test_checkpoint(multiplier_type, num_constraints, num_variables, device):
    if num_constraints > num_variables:
        pytest.skip("Overconstrained problem. Skipping test.")

    x = torch.ones(num_variables, device=device).split(1)
    model = Model(x)
    model.to(device=device)

    cmp = construct_cmp(multiplier_type, num_constraints, num_variables, device)

    primal_optimizer_class = torch.optim.SGD
    primal_optimizers = primal_optimizer_class(model.parameters(), lr=PRIMAL_LR)

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
        cmp=cmp,
        primal_optimizers=primal_optimizers,
        dual_optimizer_kwargs={"lr": DUAL_LR},
    )
    cooper_optimizer_class = type(cooper_optimizer)

    # ------------ Train the model for 100 steps ------------
    for _ in range(100):
        cooper_optimizer.roll(compute_cmp_state_kwargs=dict(x=model()))

    # Generate checkpoints after 100 steps of training
    model_state_dict_100 = model.state_dict()
    cooper_optimizer_state_dict_100 = cooper_optimizer.state_dict()
    cmp_state_dict_100 = cmp.state_dict()

    with tempfile.TemporaryDirectory() as tmpdirname:
        torch.save(model_state_dict_100, os.path.join(tmpdirname, "model.pt"))
        torch.save(cooper_optimizer_state_dict_100, os.path.join(tmpdirname, "cooper_optimizer.pt"))
        torch.save(cmp_state_dict_100, os.path.join(tmpdirname, "cmp.pt"))

        del model_state_dict_100
        del cooper_optimizer_state_dict_100
        del cmp_state_dict_100

        model_state_dict_100 = torch.load(os.path.join(tmpdirname, "model.pt"))
        cooper_optimizer_state_dict_100 = torch.load(os.path.join(tmpdirname, "cooper_optimizer.pt"))
        cmp_state_dict_100 = torch.load(os.path.join(tmpdirname, "cmp.pt"))

    # ------------ Train for *another* 100 steps ------------
    for _ in range(100):
        cooper_optimizer.roll(compute_cmp_state_kwargs=dict(x=model()))

    model_state_dict_200 = model.state_dict()
    cooper_optimizer_state_dict_200 = cooper_optimizer.state_dict()

    new_cmp = construct_cmp(multiplier_type, num_constraints, num_variables, device)
    new_cmp.load_state_dict(cmp_state_dict_100)

    x = torch.randn(num_variables, device=device).split(1)
    loaded_model = Model(x)
    loaded_model.load_state_dict(model_state_dict_100)
    loaded_model.to(device=device)

    loaded_primal_optimizers = primal_optimizer_class(loaded_model.parameters(), lr=PRIMAL_LR)

    loaded_dual_optimizers = None
    if any(new_cmp.constraints()):
        loaded_dual_optimizers = cooper_test_utils.build_dual_optimizers(
            dual_parameters=new_cmp.dual_parameters(), dual_optimizer_kwargs={"lr": DUAL_LR}
        )

    loaded_cooper_optimizer = cooper_test_utils.create_optimizer_from_kwargs(
        cooper_optimizer_class=cooper_optimizer_class,
        cmp=new_cmp,
        primal_optimizers=loaded_primal_optimizers,
        dual_optimizers=loaded_dual_optimizers,
    )
    loaded_cooper_optimizer.load_state_dict(cooper_optimizer_state_dict_100)

    # Train checkpointed model for 100 steps to reach overall 200 steps
    for _ in range(100):
        loaded_cooper_optimizer.roll(compute_cmp_state_kwargs=dict(x=loaded_model()))

    # ------------ Compare checkpoint and loaded-then-trained objects ------------
    # Compare 0-200 state_dicts versus the 0-100;100-200 state_dicts
    assert testing_utils.validate_state_dicts(loaded_model.state_dict(), model_state_dict_200)
    assert testing_utils.validate_state_dicts(loaded_cooper_optimizer.state_dict(), cooper_optimizer_state_dict_200)
