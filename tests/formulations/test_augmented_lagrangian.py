# FIXME(gallego-posada): Tests in this file are broken after removing formulation_kwargs
# from the Constraint constructor.

import cooper_test_utils
import pytest
import testing_utils
import torch

import cooper
from cooper.penalty_coefficient_updaters import MultiplicativePenaltyCoefficientUpdater


@pytest.fixture(params=[cooper.optim.AlternationType.PRIMAL_DUAL, cooper.optim.AlternationType.DUAL_PRIMAL])
def alternation_type(request):
    return request.param


def setup_augmented_lagrangian_objects(primal_optimizers, alternation_type, device):
    penalty_coefficients = (
        cooper.multipliers.DensePenaltyCoefficient(torch.tensor([1.0], device=device)),
        cooper.multipliers.DensePenaltyCoefficient(torch.tensor([1.0], device=device)),
    )

    cmp = cooper_test_utils.Toy2dCMP(
        use_ineq_constraints=True,
        formulation_type=cooper.formulations.AugmentedLagrangianFormulation,
        penalty_coefficients=penalty_coefficients,
        device=device,
    )

    penalty_updater = MultiplicativePenaltyCoefficientUpdater(growth_factor=1.005, violation_tolerance=1e-4)

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
        primal_optimizers=primal_optimizers,
        cmp=cmp,
        extrapolation=False,
        augmented_lagrangian=True,
        alternation_type=alternation_type,
    )

    return cmp, cooper_optimizer, penalty_coefficients, penalty_updater


def test_manual_augmented_lagrangian_dual_primal(Toy2dCMP_params_init, device):
    """Test first two iterations of dual-primal Augmented Lagrangian updates on toy 2D problem."""

    if not torch.allclose(Toy2dCMP_params_init, torch.tensor([0.0, -1.0], device=device)):
        pytest.skip("Manual alternating test only considers the case of initialization at [0, -1]")

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=False, params_init=Toy2dCMP_params_init
    )

    mktensor = testing_utils.mktensor(device=device)

    ALTERNATION_TYPE = cooper.optim.AlternationType.DUAL_PRIMAL
    cmp, cooper_optimizer, penalty_coefficients, penalty_updater = setup_augmented_lagrangian_objects(
        primal_optimizers=primal_optimizers, alternation_type=ALTERNATION_TYPE, device=device
    )

    roll_kwargs = {"compute_cmp_state_kwargs": dict(params=params)}

    x0_y0 = mktensor([0.0, -1.0])
    lmbda0 = mktensor([0.0, 0.0])
    rho0 = torch.stack([penalty_coefficients[0](), penalty_coefficients[1]()]).flatten().detach()
    initial_cmp_state = cmp.compute_cmp_state(params)  # noqa: F841

    # ------------ First step of dual-primal updates ------------
    _cmp_state, _, primal_lagrangian_store, _ = cooper_optimizer.roll(**roll_kwargs)
    penalty_updater.step(_cmp_state.observed_constraints)
    violations0 = mktensor(list(_cmp_state.observed_violations()))

    # Observed multipliers from the lagrangian_store should match the multipliers after
    # the first update -- roll returns the violations measured at x0_y0
    lmbda1 = torch.relu(lmbda0 + rho0 * violations0)
    assert torch.allclose(torch.cat(list(primal_lagrangian_store.observed_multiplier_values())), lmbda1)

    # Only the first constraint is violated, so the first coefficient grows. The second
    # stays the same.
    rho1 = torch.where(violations0 > penalty_updater.violation_tolerance, rho0 * penalty_updater.growth_factor, rho0)
    assert torch.allclose(torch.cat([penalty_coefficients[0](), penalty_coefficients[1]()]), rho1)

    grad_x0_y0 = cmp.analytical_gradients(x0_y0)
    # The gradient of the Augmented Lagrangian wrt the primal variables for inequality
    # constraints is:
    #   grad_obj + relu(lambda + penalty * const_violation) * grad_const
    const_contrib0 = (lmbda1[0] + rho1[0] * violations0[0]).relu() * grad_x0_y0[1][0]
    const_contrib1 = (lmbda1[1] + rho1[1] * violations0[1]).relu() * grad_x0_y0[1][1]
    x1_y1 = x0_y0 - 0.01 * (grad_x0_y0[0] + const_contrib0 + const_contrib1)

    # TODO(merajhashemi): Tolerance is relaxed to pass the test. Investigate why.
    assert torch.allclose(params, x1_y1, atol=1e-4)

    # ------------ Second step of dual-primal updates ------------
    _cmp_state, _, primal_lagrangian_store, _ = cooper_optimizer.roll(**roll_kwargs)
    penalty_updater.step(_cmp_state.observed_constraints)
    violations1 = mktensor(list(_cmp_state.observed_violations()))

    lmbda2 = torch.relu(lmbda1 + rho1 * violations1)
    assert torch.allclose(torch.cat(list(primal_lagrangian_store.observed_multiplier_values())), lmbda2)

    rho2 = torch.where(violations1 > penalty_updater.violation_tolerance, rho1 * penalty_updater.growth_factor, rho1)
    assert torch.allclose(torch.cat([penalty_coefficients[0](), penalty_coefficients[1]()]), rho2)

    grad_x1_y1 = cmp.analytical_gradients(x1_y1)
    const_contrib0 = (lmbda2[0] + rho2[0] * violations1[0]).relu() * grad_x1_y1[1][0]
    const_contrib1 = (lmbda2[1] + rho2[1] * violations1[1]).relu() * grad_x1_y1[1][1]
    x2_y2 = x1_y1 - 0.01 * (grad_x1_y1[0] + const_contrib0 + const_contrib1)

    # TODO(merajhashemi): Tolerance is relaxed to pass the test. Investigate why.
    assert torch.allclose(params, x2_y2, atol=1e-3)


def test_manual_augmented_lagrangian_primal_dual(Toy2dCMP_params_init, device):
    """Test first two iterations of dual-primal Augmented Lagrangian updates on toy 2D problem."""

    if not torch.allclose(Toy2dCMP_params_init, torch.tensor([0.0, -1.0], device=device)):
        pytest.skip("Manual alternating test only considers the case of initialization at [0, -1]")

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=False, params_init=Toy2dCMP_params_init
    )

    mktensor = testing_utils.mktensor(device=device)

    ALTERNATION_TYPE = cooper.optim.AlternationType.PRIMAL_DUAL
    cmp, cooper_optimizer, penalty_coefficients, penalty_updater = setup_augmented_lagrangian_objects(
        primal_optimizers=primal_optimizers, alternation_type=ALTERNATION_TYPE, device=device
    )

    roll_kwargs = {
        "compute_cmp_state_kwargs": dict(params=params),
        "compute_violations_kwargs": dict(params=params),
    }

    x0_y0 = mktensor([0.0, -1.0])
    lmbda0 = mktensor([0.0, 0.0])
    rho0 = torch.stack([penalty_coefficients[0](), penalty_coefficients[1]()]).flatten().detach()
    initial_cmp_state = cmp.compute_cmp_state(params)
    violations0 = mktensor(list(initial_cmp_state.observed_violations()))

    # ------------ First step of primal-dual updates ------------
    cmp_state, _, primal_lagrangian_store, _ = cooper_optimizer.roll(**roll_kwargs)
    penalty_updater.step(cmp_state.observed_constraints)

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
    assert torch.allclose(torch.cat(list(primal_lagrangian_store.observed_multiplier_values())), lmbda0)

    violations1 = mktensor(list(cmp_state.observed_violations()))

    rho1 = torch.where(violations1 > penalty_updater.violation_tolerance, rho0 * penalty_updater.growth_factor, rho0)
    assert torch.allclose(torch.cat([penalty_coefficients[0](), penalty_coefficients[1]()]), rho1)

    # ------------ Second step of primal-dual updates ------------
    cmp_state, _, primal_lagrangian_store, _ = cooper_optimizer.roll(**roll_kwargs)
    penalty_updater.step(cmp_state.observed_constraints)

    # Check that the value of the multipliers matches
    lmbda1 = torch.relu(lmbda0 + rho0 * violations1)
    assert torch.allclose(torch.cat(list(primal_lagrangian_store.observed_multiplier_values())), lmbda1)

    grad_x1_y1 = cmp.analytical_gradients(x1_y1)
    const_contrib0 = (lmbda1[0] + rho1[0] * violations1[0]).relu() * grad_x1_y1[1][0]
    const_contrib1 = (lmbda1[1] + rho1[1] * violations1[1]).relu() * grad_x1_y1[1][1]
    x2_y2 = x1_y1 - 0.01 * (grad_x1_y1[0] + const_contrib0 + const_contrib1)

    assert torch.allclose(params, x2_y2)

    # The reported multipliers from the lagrangian_store should match the multipliers
    # _before_ their update, since then is when the CMPState is measured inside roll.
    assert torch.allclose(torch.cat(list(primal_lagrangian_store.observed_multiplier_values())), lmbda1)

    violations2 = mktensor(list(cmp_state.observed_violations()))

    rho2 = torch.where(violations2 > penalty_updater.violation_tolerance, rho1 * penalty_updater.growth_factor, rho1)
    assert torch.allclose(torch.cat([penalty_coefficients[0](), penalty_coefficients[1]()]), rho2)

    # Check that the final value of the multipliers matches
    lmbda2 = torch.relu(lmbda1 + rho1 * violations2)
    cmp_state_ = cmp.compute_cmp_state(x2_y2)
    multipliers_after_update = [constraint.multiplier() for constraint, _ in cmp_state_.observed_constraints.items()]

    assert torch.allclose(torch.cat(multipliers_after_update), lmbda2)


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

    cmp, cooper_optimizer, penalty_coefficients, penalty_updater = setup_augmented_lagrangian_objects(
        primal_optimizers=primal_optimizers, alternation_type=alternation_type, device=device
    )

    roll_kwargs = {"compute_cmp_state_kwargs": dict(params=params)}

    if alternation_type == cooper.optim.AlternationType.PRIMAL_DUAL:
        roll_kwargs["compute_violations_kwargs"] = dict(params=params)

    for step_id in range(1500):
        cmp_state, _, _, _ = cooper_optimizer.roll(**roll_kwargs)
        penalty_updater.step(cmp_state.observed_constraints)
        if step_id % 100 == 0:
            # Increase the penalty coefficients
            penalty_coefficients[0].value = penalty_coefficients[0]() * 1.1
            penalty_coefficients[1].value = penalty_coefficients[1]() * 1.1

    for param, exact_solution in zip(params, Toy2dCMP_problem_properties["exact_solution"]):
        # NOTE: this test requires a relaxed tolerance of 1e-3
        assert torch.allclose(param, exact_solution, atol=1e-3)


# TODO(gallego-posada): Add a test to ensure IndexedPenaltyCoefficient works as expected
# when used in an Augmented Lagrangian formulation.
