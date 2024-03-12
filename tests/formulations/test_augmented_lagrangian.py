import os
import tempfile

import cooper_test_utils
import pytest
import testing_utils
import torch

import cooper


@pytest.fixture(params=[cooper.optim.AlternationType.PRIMAL_DUAL, cooper.optim.AlternationType.DUAL_PRIMAL])
def alternation_type(request):
    return request.param


def setup_augmented_lagrangian_objects(primal_optimizers, alternation_type, device):
    penalty_coefficients = (
        cooper.multipliers.DensePenaltyCoefficient(torch.tensor(1.0, device=device)),
        cooper.multipliers.DensePenaltyCoefficient(torch.tensor(1.0, device=device)),
    )

    formulation_kwargs = dict(penalty_growth_factor=1.005, violation_tolerance=1e-4)
    cmp = cooper_test_utils.Toy2dCMP(
        use_ineq_constraints=True,
        formulation_type=cooper.AugmentedLagrangianFormulation,
        penalty_coefficients=penalty_coefficients,
        device=device,
        formulation_kwargs=formulation_kwargs,
    )

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
        primal_optimizers=primal_optimizers,
        multipliers=cmp.multipliers,
        extrapolation=False,
        augmented_lagrangian=True,
        alternation_type=alternation_type,
    )

    return cmp, cooper_optimizer, penalty_coefficients, formulation_kwargs


def test_manual_augmented_lagrangian_dual_primal(Toy2dCMP_params_init, device):
    """Test first two iterations of dual-primal Augmented Lagrangian updates on toy 2D problem."""

    if not torch.allclose(Toy2dCMP_params_init, torch.tensor([0.0, -1.0], device=device)):
        pytest.skip("Manual alternating test only considers the case of initialization at [0, -1]")

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=False, params_init=Toy2dCMP_params_init
    )

    mktensor = testing_utils.mktensor(device=device)

    ALTERNATION_TYPE = cooper.optim.AlternationType.DUAL_PRIMAL
    cmp, cooper_optimizer, penalty_coefficients, formulation_kwargs = setup_augmented_lagrangian_objects(
        primal_optimizers=primal_optimizers, alternation_type=ALTERNATION_TYPE, device=device
    )
    penalty_growth_factor = formulation_kwargs["penalty_growth_factor"]
    violation_tolerance = formulation_kwargs["violation_tolerance"]

    roll_kwargs = {"compute_cmp_state_fn": lambda: cmp.compute_cmp_state(params)}

    x0_y0 = mktensor([0.0, -1.0])
    lmbda0 = mktensor([0.0, 0.0])
    rho0 = torch.stack([penalty_coefficients[0](), penalty_coefficients[1]()]).flatten().detach()
    initial_cmp_state = cmp.compute_cmp_state(params)  # noqa: F841

    # ------------ First step of dual-primal updates ------------
    _cmp_state, lagrangian_store = cooper_optimizer.roll(**roll_kwargs)
    violations0 = mktensor([_[1].violation for _ in _cmp_state.observed_constraints])

    # Observed multipliers from the lagrangian_store should match the multipliers after
    # the first update -- roll returns the violations measured at x0_y0
    lmbda1 = torch.relu(lmbda0 + rho0 * violations0)
    assert torch.allclose(torch.cat(lagrangian_store.multiplier_values_for_primal_constraints()), lmbda1)

    # Only the first constraint is violated, so the first coefficient grows. The second
    # stays the same.
    rho1 = torch.where(violations0 > violation_tolerance, rho0 * penalty_growth_factor, rho0).detach()
    assert torch.allclose(torch.cat([penalty_coefficients[0](), penalty_coefficients[1]()]), rho1)

    grad_x0_y0 = cmp.analytical_gradients(x0_y0)
    # The gradient of the Augmented Lagrangian wrt the primal variables for inequality
    # constraints is:
    #   grad_obj + relu(lambda + penalty * const_violation) * grad_const
    const_contrib0 = (lmbda1[0] + rho1[0] * violations0[0]).relu() * grad_x0_y0[1][0]
    const_contrib1 = (lmbda1[1] + rho1[1] * violations0[1]).relu() * grad_x0_y0[1][1]
    x1_y1 = x0_y0 - 0.01 * (grad_x0_y0[0] + const_contrib0 + const_contrib1)

    assert torch.allclose(params, x1_y1)

    # ------------ Second step of dual-primal updates ------------
    _cmp_state, lagrangian_store = cooper_optimizer.roll(**roll_kwargs)
    violations1 = mktensor([_[1].violation for _ in _cmp_state.observed_constraints])

    lmbda2 = torch.relu(lmbda1 + rho1 * violations1)
    assert torch.allclose(torch.cat(lagrangian_store.multiplier_values_for_primal_constraints()), lmbda2)

    rho2 = torch.where(violations1 > violation_tolerance, rho1 * penalty_growth_factor, rho1).detach()
    assert torch.allclose(torch.cat([penalty_coefficients[0](), penalty_coefficients[1]()]), rho2)

    grad_x1_y1 = cmp.analytical_gradients(x1_y1)
    const_contrib0 = (lmbda2[0] + rho2[0] * violations1[0]).relu() * grad_x1_y1[1][0]
    const_contrib1 = (lmbda2[1] + rho2[1] * violations1[1]).relu() * grad_x1_y1[1][1]
    x2_y2 = x1_y1 - 0.01 * (grad_x1_y1[0] + const_contrib0 + const_contrib1)

    assert torch.allclose(params, x2_y2)


def test_manual_augmented_lagrangian_primal_dual(Toy2dCMP_params_init, device):
    """Test first two iterations of dual-primal Augmented Lagrangian updates on toy 2D problem."""

    if not torch.allclose(Toy2dCMP_params_init, torch.tensor([0.0, -1.0], device=device)):
        pytest.skip("Manual alternating test only considers the case of initialization at [0, -1]")

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=False, params_init=Toy2dCMP_params_init
    )

    mktensor = testing_utils.mktensor(device=device)

    ALTERNATION_TYPE = cooper.optim.AlternationType.PRIMAL_DUAL
    cmp, cooper_optimizer, penalty_coefficients, formulation_kwargs = setup_augmented_lagrangian_objects(
        primal_optimizers=primal_optimizers, alternation_type=ALTERNATION_TYPE, device=device
    )
    penalty_growth_factor = formulation_kwargs["penalty_growth_factor"]
    violation_tolerance = formulation_kwargs["violation_tolerance"]

    roll_kwargs = {
        "compute_cmp_state_fn": lambda: cmp.compute_cmp_state(params),
        "compute_violations_fn": lambda: cmp.compute_violations(params),
    }

    x0_y0 = mktensor([0.0, -1.0])
    lmbda0 = mktensor([0.0, 0.0])
    rho0 = torch.stack([penalty_coefficients[0](), penalty_coefficients[1]()]).flatten().detach()
    initial_cmp_state = cmp.compute_cmp_state(params)
    violations0 = mktensor([_[1].violation for _ in initial_cmp_state.observed_constraints])

    # ------------ First step of primal-dual updates ------------
    _cmp_state, lagrangian_store = cooper_optimizer.roll(**roll_kwargs)
    violations1 = mktensor([_[1].violation for _ in _cmp_state.observed_constraints])

    grad_x0_y0 = cmp.analytical_gradients(x0_y0)
    # The gradient of the Augmented Lagrangian wrt the primal variables for inequality
    # constraints is:
    #   grad_obj + relu(lambda + penalty * const_violation) * grad_const
    const_contrib0 = (lmbda0[0] + rho0[0] * violations0[0]).relu() * grad_x0_y0[1][0]
    const_contrib1 = (lmbda0[1] + rho0[1] * violations0[1]).relu() * grad_x0_y0[1][1]
    x1_y1 = x0_y0 - 0.01 * (grad_x0_y0[0] + const_contrib0 + const_contrib1)

    assert torch.allclose(params, x1_y1)

    # The reported multipliers from the lagrangian_store should match the multipliers
    # _before_ their update, since then is when the CMPState is measured inside roll.
    assert torch.allclose(torch.cat(lagrangian_store.multiplier_values_for_dual_constraints()), lmbda0)

    # Check that the _current_ value of the multipliers matches
    lmbda1 = torch.relu(lmbda0 + rho0 * violations1)
    assert torch.allclose(torch.cat([_() for _ in cooper_optimizer.multipliers]), lmbda1)

    rho1 = torch.where(violations1 > violation_tolerance, rho0 * penalty_growth_factor, rho0)
    assert torch.allclose(torch.cat([penalty_coefficients[0](), penalty_coefficients[1]()]), rho1)

    # ------------ Second step of primal-dual updates ------------
    _cmp_state, lagrangian_store = cooper_optimizer.roll(**roll_kwargs)
    violations2 = mktensor([_[1].violation for _ in _cmp_state.observed_constraints])

    grad_x1_y1 = cmp.analytical_gradients(x1_y1)
    const_contrib0 = (lmbda1[0] + rho1[0] * violations1[0]).relu() * grad_x1_y1[1][0]
    const_contrib1 = (lmbda1[1] + rho1[1] * violations1[1]).relu() * grad_x1_y1[1][1]
    x2_y2 = x1_y1 - 0.01 * (grad_x1_y1[0] + const_contrib0 + const_contrib1)

    assert torch.allclose(params, x2_y2)

    # The reported multipliers from the lagrangian_store should match the multipliers
    # _before_ their update, since then is when the CMPState is measured inside roll.
    assert torch.allclose(torch.cat(lagrangian_store.multiplier_values_for_dual_constraints()), lmbda1)

    # Check that the _current_ value of the multipliers matches
    lmbda2 = torch.relu(lmbda1 + rho1 * violations2)
    assert torch.allclose(torch.cat([_() for _ in cooper_optimizer.multipliers]), lmbda2)

    rho2 = torch.where(violations2 > violation_tolerance, rho1 * penalty_growth_factor, rho1)
    assert torch.allclose(torch.cat([penalty_coefficients[0](), penalty_coefficients[1]()]), rho2)


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

    cmp, cooper_optimizer, penalty_coefficients, formulation_kwargs = setup_augmented_lagrangian_objects(
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
            penalty_coefficients[0].value = penalty_coefficients[0]() * 1.1
            penalty_coefficients[1].value = penalty_coefficients[1]() * 1.1

    for param, exact_solution in zip(params, Toy2dCMP_problem_properties["exact_solution"]):
        # NOTE: this test requires a relaxed tolerance of 1e-3
        assert torch.allclose(param, exact_solution, atol=1e-3)


def test_save_and_load_state_dict(alternation_type, Toy2dCMP_params_init, device):
    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=False, params_init=Toy2dCMP_params_init
    )

    cmp, cooper_optimizer, penalty_coefficients, formulation_kwargs = setup_augmented_lagrangian_objects(
        primal_optimizers=primal_optimizers, alternation_type=alternation_type, device=device
    )

    roll_kwargs = {"compute_cmp_state_fn": lambda: cmp.compute_cmp_state(params)}

    for _ in range(10):
        cooper_optimizer.roll(**roll_kwargs)

    # Generate checkpoints after 10 steps of training
    penalty_coefficient0_after10 = penalty_coefficients[0]().clone().detach()
    penalty_coefficient1_after10 = penalty_coefficients[1]().clone().detach()
    multiplier0_after10 = cmp.constraint_groups[0].multiplier().clone().detach()
    multiplier1_after10 = cmp.constraint_groups[1].multiplier().clone().detach()

    with tempfile.TemporaryDirectory() as tmpdirname:
        torch.save(cmp.constraint_groups[0].state_dict(), os.path.join(tmpdirname, "cg0.pt"))
        torch.save(cmp.constraint_groups[1].state_dict(), os.path.join(tmpdirname, "cg1.pt"))

        cg0_state_dict = torch.load(os.path.join(tmpdirname, "cg0.pt"))
        cg1_state_dict = torch.load(os.path.join(tmpdirname, "cg1.pt"))

    # Train for another 10 steps -- so a total of 20 steps
    for _ in range(10):
        cmp_state = cooper_optimizer.roll(**roll_kwargs)  # noqa: F841

    # Reload from checkpoint at 10 steps
    new_penalty_coefficient0 = cooper.multipliers.DensePenaltyCoefficient(torch.tensor(1.0, device=device))
    new_penalty_coefficient1 = cooper.multipliers.DensePenaltyCoefficient(torch.tensor(1.0, device=device))
    new_cmp = cooper_test_utils.Toy2dCMP(
        use_ineq_constraints=True,
        formulation_type=cooper.AugmentedLagrangianFormulation,
        penalty_coefficients=(new_penalty_coefficient0, new_penalty_coefficient1),
        device=device,
    )
    new_cmp.constraint_groups[0].load_state_dict(cg0_state_dict)
    new_cmp.constraint_groups[1].load_state_dict(cg1_state_dict)

    # Ensure loaded value of the penalty coefficient matches that observe when creating
    # the checkpoint
    new_penalty_coefficient0_value = new_penalty_coefficient0().clone().detach()
    new_penalty_coefficient1_value = new_penalty_coefficient1().clone().detach()
    assert torch.allclose(new_penalty_coefficient0_value, penalty_coefficient0_after10)
    assert torch.allclose(new_penalty_coefficient1_value, penalty_coefficient1_after10)

    # Ensure loaded value of the multipliers matches that observe when creating the
    # checkpoint
    new_multiplier0_value = new_cmp.constraint_groups[0].multiplier().clone().detach()
    new_multiplier1_value = new_cmp.constraint_groups[1].multiplier().clone().detach()
    if new_multiplier0_value != 0:
        assert not torch.allclose(new_multiplier0_value, cmp.constraint_groups[0].multiplier())
    if new_multiplier1_value != 0:
        assert not torch.allclose(new_multiplier1_value, cmp.constraint_groups[1].multiplier())
    assert torch.allclose(new_multiplier0_value, multiplier0_after10)
    assert torch.allclose(new_multiplier1_value, multiplier1_after10)


# TODO(gallego-posada): Add a test to ensure IndexedPenaltyCoefficient works as expected
# when used in an Augmented Lagrangian formulation.
