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
    if formulation_type == cooper.FormulationType.PENALTY:
        constraint_type = cooper.ConstraintType.PENALTY
    else:
        constraint_type = cooper.ConstraintType.INEQUALITY

    has_multipliers = formulation_type in [
        cooper.FormulationType.LAGRANGIAN,
        cooper.FormulationType.AUGMENTED_LAGRANGIAN,
    ]
    has_penalties = formulation_type != cooper.FormulationType.LAGRANGIAN

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=False, params_init=Toy2dCMP_params_init
    )

    alternating = cooper.optim.AlternatingType.FALSE

    const1_penalty_coefficient = cooper.multipliers.DensePenaltyCoefficient(torch.tensor(1.0, device=device))
    const2_penalty_coefficient = cooper.multipliers.DensePenaltyCoefficient(torch.tensor(1.0, device=device))
    penalty_coefficients = [const1_penalty_coefficient, const2_penalty_coefficient] if has_penalties else [None, None]

    cmp = cooper_test_utils.Toy2dCMP(
        use_ineq_constraints=True,
        formulation_type=formulation_type,
        penalty_coefficients=penalty_coefficients,
        constraint_type=constraint_type,
        device=device,
    )

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
        primal_optimizers=primal_optimizers,
        constraint_groups=cmp.constraint_groups if has_multipliers else [],
        extrapolation=False,
        alternating=alternating,
    )

    roll_kwargs = {"compute_cmp_state_fn": lambda: cmp.compute_cmp_state(params)}

    for _ in range(10):
        cooper_optimizer.roll(**roll_kwargs)
        if has_penalties:
            # Multiply the penalty coefficients by 1.01
            const1_penalty_coefficient.value = const1_penalty_coefficient() * 1.01
            const2_penalty_coefficient.value = const2_penalty_coefficient() * 1.01

    # Generate checkpoints after 10 steps of training
    if has_penalties:
        penalty1_after10 = const1_penalty_coefficient().clone().detach()
        penalty2_after10 = const2_penalty_coefficient().clone().detach()
    if has_multipliers:
        multiplier1_after10 = cmp.constraint_groups[0].multiplier().clone().detach()
        multiplier2_after10 = cmp.constraint_groups[1].multiplier().clone().detach()

    with tempfile.TemporaryDirectory() as tmpdirname:
        torch.save(cmp.constraint_groups[0].state_dict(), os.path.join(tmpdirname, "cg0.pt"))
        torch.save(cmp.constraint_groups[1].state_dict(), os.path.join(tmpdirname, "cg1.pt"))

        cg0_state_dict = torch.load(os.path.join(tmpdirname, "cg0.pt"))
        cg1_state_dict = torch.load(os.path.join(tmpdirname, "cg1.pt"))

    # Train for another 10 steps
    for _ in range(10):
        cooper_optimizer.roll(**roll_kwargs)
        if has_penalties:
            const1_penalty_coefficient.value = const1_penalty_coefficient() * 1.01
            const2_penalty_coefficient.value = const2_penalty_coefficient() * 1.01

    # Reload from checkpoint
    new_const1_penalty_coefficient = cooper.multipliers.DensePenaltyCoefficient(torch.tensor(1.0, device=device))
    new_const2_penalty_coefficient = cooper.multipliers.DensePenaltyCoefficient(torch.tensor(1.0, device=device))
    new_penalty_coefficients = (
        [new_const1_penalty_coefficient, new_const2_penalty_coefficient] if has_penalties else [None, None]
    )

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
        new_penalty1_value = new_const1_penalty_coefficient().clone().detach()
        new_penalty2_value = new_const2_penalty_coefficient().clone().detach()
        assert not torch.allclose(new_penalty1_value, const1_penalty_coefficient())
        assert not torch.allclose(new_penalty2_value, const2_penalty_coefficient())

        # They should, however, be the same as the ones recorded before the checkpoint
        assert torch.allclose(new_penalty1_value, penalty1_after10)
        assert torch.allclose(new_penalty2_value, penalty2_after10)

    if has_multipliers:
        # Similar story for the multipliers
        new_multiplier1_value = new_cmp.constraint_groups[0].multiplier().clone().detach()
        new_multiplier2_value = new_cmp.constraint_groups[1].multiplier().clone().detach()

        # Ignoring the case where the multiplier is 0 as both may match simply because
        # the run is feasible
        if new_multiplier1_value != 0:
            assert not torch.allclose(new_multiplier1_value, cmp.constraint_groups[0].multiplier())
        if new_multiplier2_value != 0:
            assert not torch.allclose(new_multiplier2_value, cmp.constraint_groups[1].multiplier())
        assert torch.allclose(new_multiplier1_value, multiplier1_after10)
        assert torch.allclose(new_multiplier2_value, multiplier2_after10)
