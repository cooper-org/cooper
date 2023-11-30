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
        primal_optimizers,
        multipliers=cmp.multipliers,
        dual_optimizer_class=torch.optim.SGD,
        dual_optimizer_kwargs={"lr": 1e-2},
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

    if len(new_cmp.multipliers) == 0:
        loaded_dual_optimizers = None
    else:
        loaded_dual_optimizers = cooper_test_utils.build_dual_optimizers(multipliers=new_cmp.multipliers)

    loaded_constrained_optimizer = cooper.optim.utils.load_cooper_optimizer_from_state_dict(
        cooper_optimizer_state=constrained_optimizer_state_dict_100,
        primal_optimizers=loaded_primal_optimizers,
        dual_optimizers=loaded_dual_optimizers,
        multipliers=new_cmp.multipliers,
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


@pytest.mark.parametrize(
    "formulation_type",
    [
        cooper.FormulationType.PENALTY,
        cooper.FormulationType.QUADRATIC_PENALTY,
        cooper.FormulationType.LAGRANGIAN,
        cooper.FormulationType.AUGMENTED_LAGRANGIAN,
    ],
)
def test_formulation_checkpoint(formulation_type, Toy2dCMP_params_init, device):
    formulation_class = formulation_type.value

    if formulation_type == cooper.FormulationType.PENALTY:
        constraint_type = cooper.ConstraintType.PENALTY
    else:
        constraint_type = cooper.ConstraintType.INEQUALITY

    has_multiplier = formulation_class.expects_multiplier
    has_penalties = formulation_class.expects_penalty_coefficient

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=False, params_init=Toy2dCMP_params_init
    )

    def make_fresh_penalty_coefficients(has_penalties):
        if has_penalties:
            penalty_coefficient0 = cooper.multipliers.DensePenaltyCoefficient(torch.tensor(1.0, device=device))
            penalty_coefficient1 = cooper.multipliers.DensePenaltyCoefficient(torch.tensor(1.0, device=device))
            return [penalty_coefficient0, penalty_coefficient1]
        else:
            return None

    penalty_coefficients = make_fresh_penalty_coefficients(has_penalties=has_penalties)
    cmp = cooper_test_utils.Toy2dCMP(
        use_ineq_constraints=True,
        formulation_type=formulation_type,
        penalty_coefficients=penalty_coefficients,
        constraint_type=constraint_type,
        device=device,
    )

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
        primal_optimizers=primal_optimizers,
        multipliers=cmp.multipliers,
        extrapolation=False,
        alternation_type=cooper.optim.AlternationType.FALSE,
    )

    roll_kwargs = {"compute_cmp_state_fn": lambda: cmp.compute_cmp_state(params)}

    for _ in range(10):
        cooper_optimizer.roll(**roll_kwargs)
        if has_penalties:
            # Multiply the penalty coefficients by 1.01
            for _penalty_coefficient in penalty_coefficients:
                _penalty_coefficient.value = _penalty_coefficient() * 1.01

    # Generate checkpoints after 10 steps of training
    if has_penalties:
        penalty_coefficient0_after10 = penalty_coefficients[0]().clone().detach()
        penalty_coefficient1_after10 = penalty_coefficients[1]().clone().detach()

    if has_multiplier:
        multiplier0_after10 = cmp.constraint_groups[0].multiplier().clone().detach()
        multiplier1_after10 = cmp.constraint_groups[1].multiplier().clone().detach()

    with tempfile.TemporaryDirectory() as tmpdirname:
        torch.save(cmp.constraint_groups[0].state_dict(), os.path.join(tmpdirname, "cg0.pt"))
        torch.save(cmp.constraint_groups[1].state_dict(), os.path.join(tmpdirname, "cg1.pt"))

        cg0_state_dict = torch.load(os.path.join(tmpdirname, "cg0.pt"))
        cg1_state_dict = torch.load(os.path.join(tmpdirname, "cg1.pt"))

    # Train for another 10 steps
    for _ in range(10):
        cooper_optimizer.roll(**roll_kwargs)
        if has_penalties:
            # Multiply the penalty coefficients by 1.01
            for penalty_coefficient in penalty_coefficients:
                penalty_coefficient.value = penalty_coefficient() * 1.01

    # Reload from checkpoint
    new_penalty_coefficients = make_fresh_penalty_coefficients(has_penalties=has_penalties)

    new_cmp = cooper_test_utils.Toy2dCMP(
        use_ineq_constraints=True,
        formulation_type=formulation_type,
        penalty_coefficients=new_penalty_coefficients,
        constraint_type=constraint_type,
        device=device,
    )
    new_cmp.constraint_groups[0].load_state_dict(cg0_state_dict)
    new_cmp.constraint_groups[1].load_state_dict(cg1_state_dict)

    if has_penalties:
        # The loaded penalty coefficients come from 10 steps of training, so they should be
        # different from the current ones
        new_penalty_coefficient0_value = new_penalty_coefficients[0]().clone().detach()
        new_penalty_coefficient1_value = new_penalty_coefficients[1]().clone().detach()
        assert not torch.allclose(new_penalty_coefficient0_value, penalty_coefficients[0]())
        assert not torch.allclose(new_penalty_coefficient1_value, penalty_coefficients[1]())

        # They should, however, be the same as the ones recorded before the checkpoint
        assert torch.allclose(new_penalty_coefficient0_value, penalty_coefficient0_after10)
        assert torch.allclose(new_penalty_coefficient1_value, penalty_coefficient1_after10)

    if has_multiplier:
        # Similar story for the multipliers
        new_multiplier0_value = new_cmp.constraint_groups[0].multiplier().clone().detach()
        new_multiplier1_value = new_cmp.constraint_groups[1].multiplier().clone().detach()

        # Ignoring the case where the multiplier is 0 as both may match simply because
        # the run is feasible
        if new_multiplier0_value != 0:
            assert not torch.allclose(new_multiplier0_value, cmp.constraint_groups[0].multiplier())
        if new_multiplier1_value != 0:
            assert not torch.allclose(new_multiplier1_value, cmp.constraint_groups[1].multiplier())
        assert torch.allclose(new_multiplier0_value, multiplier0_after10)
        assert torch.allclose(new_multiplier1_value, multiplier1_after10)
