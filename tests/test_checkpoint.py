#!/usr/bin/env python

"""Tests for checkpointing of constrained and unconstrained experiments."""

import os
import tempfile

# Import basic closure example from helpers
import cooper_test_utils
import pytest
import torch

import cooper


class Model(torch.nn.Module):
    def __init__(self, params: list):
        super(Model, self).__init__()

        self.num_params = len(params)

        for i, param in enumerate(params):
            if not isinstance(param, torch.nn.Parameter):
                param = torch.nn.Parameter(param)
            self.register_parameter(name=f"params_{i}", param=param)

    def forward(self):
        return torch.stack([getattr(self, f"params_{i}") for i in range(self.num_params)])


def test_checkpoint(Toy2dCMP_problem_properties, Toy2dCMP_params_init, use_multiple_primal_optimizers, device):
    """Test checkpointing and loading for constrained and unconstrained cases."""

    use_ineq_constraints = Toy2dCMP_problem_properties["use_ineq_constraints"]

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers, Toy2dCMP_params_init
    )
    model = Model(params)

    cmp = cooper_test_utils.Toy2dCMP(use_ineq_constraints=use_ineq_constraints, device=device)

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
        primal_optimizers, cmp.constraint_groups, dual_optimizer_name="SGD", dual_optimizer_kwargs={"lr": 1e-2}
    )

    compute_cmp_state_fn = lambda: cmp.compute_cmp_state(model)

    # ------------ Train the model for 100 steps ------------
    for _ in range(100):
        cmp_state, lagrangian_store = cooper_optimizer.roll(compute_cmp_state_fn)

    # Generate checkpoints after 100 steps of training
    model_state_dict_100 = model.state_dict()
    constrained_optimizer_state_dict_100 = cooper_optimizer.state_dict()

    with tempfile.TemporaryDirectory() as tmpdirname:
        torch.save(model_state_dict_100, os.path.join(tmpdirname, "model.pt"))
        torch.save(constrained_optimizer_state_dict_100, os.path.join(tmpdirname, "constrained_optimizer.pt"))

        del model_state_dict_100
        del constrained_optimizer_state_dict_100

        model_state_dict_100 = torch.load(os.path.join(tmpdirname, "model.pt"))
        constrained_optimizer_state_dict_100 = torch.load(os.path.join(tmpdirname, "constrained_optimizer.pt"))

    # ------------ Train for *another* 100 steps ------------
    for _ in range(100):
        cmp_state, lagrangian_store = cooper_optimizer.roll(compute_cmp_state_fn)

    model_state_dict_200 = model.state_dict()
    constrained_optimizer_state_dict_200 = cooper_optimizer.state_dict()

    # ------------ Reload from 100-step checkpoint ------------
    new_cmp = cooper_test_utils.Toy2dCMP(use_ineq_constraints=use_ineq_constraints, device=device)

    loaded_params, loaded_primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers, Toy2dCMP_params_init
    )
    loaded_model = Model(params=loaded_params)
    loaded_model.load_state_dict(model_state_dict_100)
    loaded_model.to(device)

    loaded_dual_optimizers = cooper_test_utils.build_dual_optimizers(
        is_constrained=use_ineq_constraints, constraint_groups=new_cmp.constraint_groups
    )

    loaded_constrained_optimizer = cooper.optim.utils.load_cooper_optimizer_from_state_dict(
        cooper_optimizer_state=constrained_optimizer_state_dict_100,
        primal_optimizers=loaded_primal_optimizers,
        dual_optimizers=loaded_dual_optimizers,
        constraint_groups=new_cmp.constraint_groups,
    )

    # Train checkpointed model for 100 steps to reach overall 200 steps
    compute_cmp_state_fn = lambda: new_cmp.compute_cmp_state(loaded_model)
    for _ in range(100):
        cmp_state, lagrangian_store = loaded_constrained_optimizer.roll(compute_cmp_state_fn)

    # ------------ Compare checkpoint and loaded-then-trained objects ------------
    # Compare 0-200 state_dicts versus the 0-100;100-200 state_dicts
    assert cooper.utils.validate_state_dicts(loaded_model.state_dict(), model_state_dict_200)
    # These are ConstrainedOptimizerState objects and not dicts
    assert loaded_constrained_optimizer.state_dict() == constrained_optimizer_state_dict_200
