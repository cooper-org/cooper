"""Tests for checkpointing of constrained and unconstrained experiments."""

import os
import tempfile
from collections.abc import Sequence

import torch

import testing

DUAL_LR = 1e-2


class Model(torch.nn.Module):
    """A simple model that concatenates a list of parameters."""

    def __init__(self, params: Sequence[torch.Tensor]):
        super().__init__()
        self.num_params = len(params)

        for i, param in enumerate(params):
            self.register_parameter(name=f"params_{i}", param=torch.nn.Parameter(param))

    def forward(self):
        return torch.cat([getattr(self, f"params_{i}") for i in range(self.num_params)])


def construct_cmp(multiplier_type, penalty_coefficient_type, formulation_type, num_constraints, num_variables, device):
    generator = torch.Generator(device).manual_seed(0)
    A = torch.randn(num_constraints, num_variables, device=device, generator=generator)
    b = torch.randn(num_constraints, device=device, generator=generator)

    return testing.SquaredNormLinearCMP(
        num_variables=num_variables,
        has_ineq_constraint=True,
        ineq_multiplier_type=multiplier_type,
        ineq_penalty_coefficient_type=penalty_coefficient_type,
        ineq_formulation_type=formulation_type,
        A=A,
        b=b,
        device=device,
    )


def test_checkpoint(
    multiplier_type,
    penalty_coefficient_type,
    formulation_type,
    use_multiple_primal_optimizers,
    num_constraints,
    num_variables,
    device,
):
    x = [torch.ones(num_variables, device=device)]
    if use_multiple_primal_optimizers:
        x = x[0].split(1)
    model = Model(x)
    model.to(device=device)

    cmp = construct_cmp(
        multiplier_type, penalty_coefficient_type, formulation_type, num_constraints, num_variables, device
    )

    primal_optimizers = testing.build_primal_optimizers(list(model.parameters()))
    cooper_optimizer = testing.build_cooper_optimizer(
        cmp=cmp, primal_optimizers=primal_optimizers, dual_optimizer_kwargs={"lr": DUAL_LR}
    )
    cooper_optimizer_class = type(cooper_optimizer)

    # ------------ Train the model for 100 steps ------------
    for _ in range(100):
        cooper_optimizer.roll(compute_cmp_state_kwargs={"x": model()})

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

        model_state_dict_100 = torch.load(os.path.join(tmpdirname, "model.pt"), weights_only=True)
        cooper_optimizer_state_dict_100 = torch.load(os.path.join(tmpdirname, "cooper_optimizer.pt"), weights_only=True)
        cmp_state_dict_100 = torch.load(os.path.join(tmpdirname, "cmp.pt"), weights_only=True)

    # ------------ Train for *another* 100 steps ------------
    for _ in range(100):
        cooper_optimizer.roll(compute_cmp_state_kwargs={"x": model()})

    model_state_dict_200 = model.state_dict()
    cooper_optimizer_state_dict_200 = cooper_optimizer.state_dict()
    cmp_state_dict_200 = cmp.state_dict()

    # ------------ Reload from 100-step checkpoint ------------
    new_cmp = construct_cmp(
        multiplier_type, penalty_coefficient_type, formulation_type, num_constraints, num_variables, device
    )
    new_cmp.load_state_dict(cmp_state_dict_100)

    x = [torch.randn(num_variables, device=device)]
    if use_multiple_primal_optimizers:
        x = x[0].split(1)
    loaded_model = Model(x)
    loaded_model.load_state_dict(model_state_dict_100)
    loaded_model.to(device=device)

    loaded_primal_optimizers = testing.build_primal_optimizers(list(loaded_model.parameters()))
    loaded_dual_optimizers = None
    if any(True for _ in new_cmp.dual_parameters()):
        loaded_dual_optimizers = testing.build_dual_optimizer(
            dual_parameters=new_cmp.dual_parameters(), dual_optimizer_kwargs={"lr": DUAL_LR}
        )

    loaded_cooper_optimizer = cooper_optimizer_class(
        cmp=new_cmp, primal_optimizers=loaded_primal_optimizers, dual_optimizers=loaded_dual_optimizers
    )
    loaded_cooper_optimizer.load_state_dict(cooper_optimizer_state_dict_100)

    # Train checkpointed model for 100 steps to reach overall 200 steps
    for _ in range(100):
        loaded_cooper_optimizer.roll(compute_cmp_state_kwargs={"x": loaded_model()})

    # ------------ Compare checkpoint and loaded-then-trained objects ------------
    # Compare 0-200 state_dicts versus the 0-100;100-200 state_dicts
    assert testing.validate_state_dicts(loaded_model.state_dict(), model_state_dict_200)
    assert testing.validate_state_dicts(loaded_cooper_optimizer.state_dict(), cooper_optimizer_state_dict_200)
    assert testing.validate_state_dicts(new_cmp.state_dict(), cmp_state_dict_200)
