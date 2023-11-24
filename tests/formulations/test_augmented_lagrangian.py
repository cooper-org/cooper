import os
import tempfile

import cooper_test_utils
import pytest
import testing_utils
import torch

import cooper


def setup_augmented_lagrangian_objects(primal_optimizers, alternation_type, device):
    penalty_coefficient0 = cooper.multipliers.DensePenaltyCoefficient(torch.tensor(1.0, device=device))
    penalty_coefficient1 = cooper.multipliers.DensePenaltyCoefficient(torch.tensor(1.0, device=device))

    cmp = cooper_test_utils.Toy2dCMP(
        use_ineq_constraints=True,
        formulation_type=cooper.FormulationType.AUGMENTED_LAGRANGIAN,
        penalty_coefficients=(penalty_coefficient0, penalty_coefficient1),
        device=device,
    )

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
        primal_optimizers=primal_optimizers,
        multipliers=cmp.multipliers,
        extrapolation=False,
        alternation_type=alternation_type,
    )

    return cmp, cooper_optimizer, penalty_coefficient0, penalty_coefficient1


def test_manual_augmented_lagrangian_simultaneous(Toy2dCMP_params_init, device):
    """Test first two iterations of simultaneous GDA updates on toy 2D problem."""

    if not torch.allclose(Toy2dCMP_params_init, torch.tensor([0.0, -1.0], device=device)):
        pytest.skip("Manual alternating test only considers the case of initialization at [0, -1]")

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=False, params_init=Toy2dCMP_params_init
    )

    mktensor = testing_utils.mktensor(device=device)

    cmp, cooper_optimizer, penalty_coefficient0, penalty_coefficient1 = setup_augmented_lagrangian_objects(
        primal_optimizers=primal_optimizers, alternation_type=cooper.optim.AlternationType.FALSE, device=device
    )

    roll_kwargs = {"compute_cmp_state_fn": lambda: cmp.compute_cmp_state(params)}

    x0_y0 = mktensor([0.0, -1.0])
    lmbda0 = mktensor([0.0, 0.0])

    # ------------ First step of simultaneous updates ------------
    _cmp_state, lagrangian_store = cooper_optimizer.roll(**roll_kwargs)
    violations = mktensor([_[1].violation for _ in _cmp_state.observed_constraints])

    # Observed multipliers from the lagrangian_store should be before the update, thus
    # matching lmbda0
    assert torch.allclose(torch.cat(lagrangian_store.multiplier_values_for_primal_constraints()), lmbda0)

    grad_x0_y0 = cmp.analytical_gradients(x0_y0)
    # The gradient of the Augmented Lagrangian wrt the primal variables is:
    # grad_obj + (lambda + penalty * const_violation) * grad_const
    # Both constraints in the Toy2dCMP problem are inequality constraints, so relu
    const_contrib0 = (lmbda0[0] + penalty_coefficient0() * violations[0].relu()) * grad_x0_y0[1][0]
    const_contrib1 = (lmbda0[1] + penalty_coefficient1() * violations[1].relu()) * grad_x0_y0[1][1]
    x1_y1 = x0_y0 - 1e-2 * (grad_x0_y0[0] + const_contrib0 + const_contrib1)

    assert torch.allclose(params, x1_y1)

    # The update to the multipliers has the learning rate and the penalty coefficient
    lmbda_update = torch.cat([violations[0] * penalty_coefficient0(), violations[1] * penalty_coefficient1()])
    lmbda1 = torch.relu(lmbda0 + 1e-2 * lmbda_update)

    # Increase the penalty coefficients
    penalty_coefficient0.value = penalty_coefficient0.value * 2
    penalty_coefficient1.value = penalty_coefficient1.value * 2

    # ------------ Second step of simultaneous updates ------------
    _cmp_state, lagrangian_store = cooper_optimizer.roll(**roll_kwargs)
    violations = mktensor([_[1].violation for _ in _cmp_state.observed_constraints])

    assert torch.allclose(torch.cat(lagrangian_store.multiplier_values_for_primal_constraints()), lmbda1)

    grad_x1_y1 = cmp.analytical_gradients(x1_y1)
    const_contrib0 = (lmbda1[0] + penalty_coefficient0() * violations[0].relu()) * grad_x1_y1[1][0]
    const_contrib1 = (lmbda1[1] + penalty_coefficient1() * violations[1].relu()) * grad_x1_y1[1][1]
    x2_y2 = x1_y1 - 1e-2 * (grad_x1_y1[0] + const_contrib0 + const_contrib1)

    assert torch.allclose(params, x2_y2)


def test_manual_augmented_lagrangian_primal_dual():
    pass


def test_manual_augmented_lagrangian_dual_primal():
    pass


@pytest.mark.parametrize(
    "alternation_type", [cooper.optim.AlternationType.PRIMAL_DUAL, cooper.optim.AlternationType.DUAL_PRIMAL, False]
)
def test_convergence_augmented_lagrangian(
    alternation_type, Toy2dCMP_params_init, Toy2dCMP_problem_properties, use_multiple_primal_optimizers, device
):
    """Test convergence of Augmented Lagrangian updates on toy 2D problem."""

    use_ineq_constraints = Toy2dCMP_problem_properties["use_ineq_constraints"]
    if not use_ineq_constraints:
        pytest.skip("Alternating updates requires a problem with constraints.")

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=use_multiple_primal_optimizers, params_init=Toy2dCMP_params_init
    )

    cmp, cooper_optimizer, penalty_coefficient0, penalty_coefficient1 = setup_augmented_lagrangian_objects(
        primal_optimizers=primal_optimizers, alternation_type=alternation_type, device=device
    )

    compute_cmp_state_fn = lambda: cmp.compute_cmp_state(params)
    roll_kwargs = {"compute_cmp_state_fn": compute_cmp_state_fn}

    if alternation_type == cooper.optim.AlternationType.PRIMAL_DUAL:
        roll_kwargs["compute_violations_fn"] = lambda: cmp.compute_violations(params)

    for step_id in range(1500):
        cooper_optimizer.roll(**roll_kwargs)
        if step_id % 100 == 0:
            # Increase the penalty coefficients
            penalty_coefficient0.value = penalty_coefficient0() * 1.1
            penalty_coefficient1.value = penalty_coefficient1() * 1.1

    for param, exact_solution in zip(params, Toy2dCMP_problem_properties["exact_solution"]):
        # NOTE: this test requires a relaxed tolerance of 1e-4
        assert torch.allclose(param, exact_solution, atol=1e-4)


def test_save_and_load_state_dict(Toy2dCMP_params_init, device):
    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=False, params_init=Toy2dCMP_params_init
    )

    cmp, cooper_optimizer, penalty_coefficient0, penalty_coefficient1 = setup_augmented_lagrangian_objects(
        primal_optimizers=primal_optimizers, alternation_type=cooper.optim.AlternationType.FALSE, device=device
    )

    roll_kwargs = {"compute_cmp_state_fn": lambda: cmp.compute_cmp_state(params)}

    for _ in range(10):
        cooper_optimizer.roll(**roll_kwargs)
        # Multiply the penalty coefficients by 1.01
        penalty_coefficient0.value = penalty_coefficient0() * 1.01
        penalty_coefficient1.value = penalty_coefficient1() * 1.01

    # Generate checkpoints after 10 steps of training
    penalty_coefficient0_after10 = penalty_coefficient0().clone().detach()
    penalty_coefficient1_after10 = penalty_coefficient1().clone().detach()
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
        penalty_coefficient0.value = penalty_coefficient0() * 1.01
        penalty_coefficient1.value = penalty_coefficient1() * 1.01

    # Reload from checkpoint
    new_penalty_coefficient0 = cooper.multipliers.DensePenaltyCoefficient(torch.tensor(1.0, device=device))
    new_penalty_coefficient1 = cooper.multipliers.DensePenaltyCoefficient(torch.tensor(1.0, device=device))
    new_cmp = cooper_test_utils.Toy2dCMP(
        use_ineq_constraints=True,
        formulation_type=cooper.FormulationType.AUGMENTED_LAGRANGIAN,
        penalty_coefficients=(new_penalty_coefficient0, new_penalty_coefficient1),
        device=device,
    )
    new_cmp.constraint_groups[0].load_state_dict(cg0_state_dict)
    new_cmp.constraint_groups[1].load_state_dict(cg1_state_dict)

    # The loaded penalty coefficients come from 10 steps of training, so they should be
    # different from the current ones
    new_penalty_coefficient0_value = new_penalty_coefficient0().clone().detach()
    new_penalty_coefficient1_value = new_penalty_coefficient1().clone().detach()
    assert not torch.allclose(new_penalty_coefficient0_value, penalty_coefficient0())
    assert not torch.allclose(new_penalty_coefficient1_value, penalty_coefficient1())

    # They should, however, be the same as the ones recorded before the checkpoint
    assert torch.allclose(new_penalty_coefficient0_value, penalty_coefficient0_after10)
    assert torch.allclose(new_penalty_coefficient1_value, penalty_coefficient1_after10)

    # Similar story for the multipliers
    new_multiplier0_value = new_cmp.constraint_groups[0].multiplier().clone().detach()
    new_multiplier1_value = new_cmp.constraint_groups[1].multiplier().clone().detach()

    if new_multiplier0_value != 0:
        assert not torch.allclose(new_multiplier0_value, cmp.constraint_groups[0].multiplier())
    if new_multiplier1_value != 0:
        assert not torch.allclose(new_multiplier1_value, cmp.constraint_groups[1].multiplier())
    assert torch.allclose(new_multiplier0_value, multiplier0_after10)
    assert torch.allclose(new_multiplier1_value, multiplier1_after10)
