import cooper_test_utils
import pytest
import testing_utils
import torch

import cooper


@pytest.mark.parametrize("use_violation_fn", [True, False])
def test_manual_primal_dual(use_violation_fn, Toy2dCMP_problem_properties, Toy2dCMP_params_init, device):
    """Test first two iterations of PrimalDual alternating GDA updates on toy 2D problem."""

    use_ineq_constraints = Toy2dCMP_problem_properties["use_ineq_constraints"]
    if not use_ineq_constraints:
        pytest.skip("Alternating updates requires a problem with constraints.")

    if not torch.allclose(Toy2dCMP_params_init, torch.tensor([0.0, -1.0], device=device)):
        pytest.skip("Manual alternating test only considers the case of initialization at [0, -1]")

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=False, params_init=Toy2dCMP_params_init
    )

    # Only perfoming this test for the case of a single primal optimizer
    assert isinstance(params, torch.nn.Parameter)

    cmp = cooper_test_utils.Toy2dCMP(use_ineq_constraints=use_ineq_constraints, device=device)

    mktensor = testing_utils.mktensor(device=device)

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
        primal_optimizers=primal_optimizers,
        cmp=cmp,
        extrapolation=False,
        alternation_type=cooper.optim.AlternationType.PRIMAL_DUAL,
        dual_optimizer_class=torch.optim.SGD,
        dual_optimizer_kwargs={"lr": 1e-2},
    )

    roll_kwargs = {
        "compute_cmp_state_kwargs": dict(params=params),
        "compute_violations_kwargs": dict(params=params) if use_violation_fn else {},
    }

    x0_y0 = mktensor([0.0, -1.0])  # noqa: F841
    lmbda0 = mktensor([0.0, 0.0])

    # ------------ First step of alternating updates ------------
    _cmp_state, primal_lagrangian_store, _ = cooper_optimizer.roll(**roll_kwargs)

    x1_y1 = mktensor([0.0, -0.96])
    assert torch.allclose(params, x1_y1)

    cmp_state = cmp.compute_cmp_state(x1_y1)

    violations = mktensor([_[1].violation for _ in cmp_state.observed_constraints])
    lmbda1 = torch.relu(lmbda0 + 1e-2 * violations)

    multiplier_values = [constraint.multiplier() for constraint, _ in cmp_state.observed_constraints]
    for multiplier_value, target_value in zip(multiplier_values, lmbda1):
        assert torch.allclose(multiplier_value, mktensor([target_value]))

    if use_violation_fn:
        # Use the loss in the CMP state before the primal update, but the violations
        # after the primal update
        lagrangian1 = _cmp_state.loss + torch.sum(violations * lmbda0)
        # When the final Lagrangian is evaluated, the primal variables have changed,
        # but the multipliers are still zero (not yet updated)
        assert torch.allclose(primal_lagrangian_store.lagrangian, lagrangian1)
    else:
        lagrangian1 = cmp_state.loss + torch.sum(violations * lmbda0)
        # Since the multipliers are still zero, the Lagrangian matches the loss at
        # the updated primal point
        assert torch.allclose(primal_lagrangian_store.lagrangian, lagrangian1)

    # ------------ Second step of alternating updates ------------
    _cmp_state, primal_lagrangian_store, _ = cooper_optimizer.roll(**roll_kwargs)

    x2_y2 = mktensor([0.0196 * 0.01, -0.96 + 3.8596 * 1e-2])
    assert torch.allclose(params, x2_y2)

    cmp_state = cmp.compute_cmp_state(x2_y2)

    violations = mktensor([_[1].violation for _ in cmp_state.observed_constraints])
    lmbda2 = torch.relu(lmbda1 + 1e-2 * violations)

    multiplier_values = [constraint.multiplier() for constraint, _ in cmp_state.observed_constraints]
    for multiplier, target_value in zip(multiplier_values, lmbda2):
        assert torch.allclose(multiplier, mktensor([target_value]))

    if use_violation_fn:
        # Use the loss in the CMP state before the primal update, but the violations
        # after the primal update
        lagrangian2 = _cmp_state.loss + torch.sum(violations * lmbda1)
        # When the final Lagrangian is evaluated, the primal variables have changed,
        # but the multipliers are still zero (not yet updated)
        assert torch.allclose(primal_lagrangian_store.lagrangian, lagrangian2)
    else:
        lagrangian2 = cmp_state.loss + torch.sum(violations * lmbda1)
        # Since the multipliers are still zero, the Lagrangian matches the loss at
        # the updated primal point
        assert torch.allclose(primal_lagrangian_store.lagrangian, lagrangian2)


def test_manual_dual_primal(Toy2dCMP_problem_properties, Toy2dCMP_params_init, device):
    """Test first two iterations of DualPrimal alternating GDA updates on toy 2D problem."""

    use_ineq_constraints = Toy2dCMP_problem_properties["use_ineq_constraints"]
    if not use_ineq_constraints:
        pytest.skip("Alternating updates requires a problem with constraints.")

    if not torch.allclose(Toy2dCMP_params_init, torch.tensor([0.0, -1.0], device=device)):
        pytest.skip("Manual alternating test only considers the case of initialization at [0, -1]")

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=False, params_init=Toy2dCMP_params_init
    )

    # Only performing this test for the case of a single primal optimizer
    assert isinstance(params, torch.nn.Parameter)

    cmp = cooper_test_utils.Toy2dCMP(use_ineq_constraints=use_ineq_constraints, device=device)

    mktensor = testing_utils.mktensor(device=device)

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
        primal_optimizers=primal_optimizers,
        cmp=cmp,
        extrapolation=False,
        alternation_type=cooper.optim.AlternationType.DUAL_PRIMAL,
        dual_optimizer_class=torch.optim.SGD,
        dual_optimizer_kwargs={"lr": 1e-2},
    )

    roll_kwargs = {"compute_cmp_state_kwargs": dict(params=params)}

    x0_y0 = mktensor([0.0, -1.0])
    lmbda0 = mktensor([0.0, 0.0])

    # ------------ First step of alternating updates ------------
    _cmp_state, lagrangian_store, _ = cooper_optimizer.roll(**roll_kwargs)

    cmp_state = cmp.compute_cmp_state(x0_y0)

    violations = mktensor([_[1].violation for _ in cmp_state.observed_constraints])
    lmbda1 = torch.relu(lmbda0 + 1e-2 * violations)

    multiplier_values = [constraint.multiplier() for constraint, _ in cmp_state.observed_constraints]
    for multiplier, target_value in zip(multiplier_values, lmbda1):
        assert torch.allclose(multiplier, mktensor([target_value]))

    x1_y1 = mktensor([0.0002, -0.9598])
    assert torch.allclose(params, x1_y1)

    # In the "DualPrimal" case, the CMPState is evaluated only once, and the
    # `compute_violations_fn` is not used.
    # Original loss = 2 ---  Original violations = [2, -2]
    # Updated multipliers = [0.02, 0.0]
    # Lagrangian value = 2 + 0.02 * 2 + 0.0 * (-2) = 2.04
    lagrangian1 = cmp_state.loss + torch.sum(violations * lmbda1)
    assert torch.allclose(lagrangian_store.lagrangian, lagrangian1)

    # ------------ Second step of alternating updates ------------
    _cmp_state, lagrangian_store, _ = cooper_optimizer.roll(**roll_kwargs)

    # loss = 1.84243212
    cmp_state = cmp.compute_cmp_state(x1_y1)

    # violation[0] = -2e-4 + 0.9598 + 1 = 1.9596
    # violation[1] = (0.0002 ** 2) - 0.9598 - 1 = -1.95979996
    violations = mktensor([_[1].violation for _ in cmp_state.observed_constraints])

    # Updated multipliers = [0.02 + 0.01 * 1.9596 = 0.039596, 0.0]
    lmbda2 = torch.relu(lmbda1 + 1e-2 * violations)
    multiplier_values = [constraint.multiplier() for constraint, _ in cmp_state.observed_constraints]
    for multiplier, target_value in zip(multiplier_values, lmbda2):
        assert torch.allclose(multiplier, mktensor([target_value]))

    # Lagrangian value = 1.84243212 + 0.039596 * 1.9596 + 0.0 * (-1.95979996) = 1.9216
    lagrangian2 = cmp_state.loss + torch.sum(violations * lmbda2)
    assert torch.allclose(lagrangian_store.lagrangian, lagrangian2)

    # grad_x = -0.039196, grad_y = -3.878796
    # x2 = 0.0002 - 0.01 * (-0.039196) = 0.00059196
    # y2 = -0.9598 - 0.01 * (-3.878796) = -0.92101204
    x2_y2 = mktensor([0.00059196, -0.92101204])
    assert torch.allclose(params, x2_y2)


@pytest.mark.parametrize(
    "alternation_type", [cooper.optim.AlternationType.PRIMAL_DUAL, cooper.optim.AlternationType.DUAL_PRIMAL]
)
@pytest.mark.parametrize("use_defect_fn", [True, False])
def test_convergence_alternating(
    alternation_type,
    use_defect_fn,
    Toy2dCMP_problem_properties,
    Toy2dCMP_params_init,
    use_multiple_primal_optimizers,
    device,
):
    """Test convergence of alternating GDA updates on toy 2D problem."""

    use_ineq_constraints = Toy2dCMP_problem_properties["use_ineq_constraints"]
    if not use_ineq_constraints:
        pytest.skip("Alternating updates requires a problem with constraints.")

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=use_multiple_primal_optimizers, params_init=Toy2dCMP_params_init
    )

    cmp = cooper_test_utils.Toy2dCMP(use_ineq_constraints=use_ineq_constraints, device=device)

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
        primal_optimizers=primal_optimizers, cmp=cmp, extrapolation=False, alternation_type=alternation_type
    )

    roll_kwargs = {"compute_cmp_state_kwargs": dict(params=params)}
    if alternation_type == cooper.optim.AlternationType.PRIMAL_DUAL:
        roll_kwargs["compute_violations_kwargs"] = dict(params=params) if use_defect_fn else {}

    for step_id in range(1500):
        cooper_optimizer.roll(**roll_kwargs)

    for param, exact_solution in zip(params, Toy2dCMP_problem_properties["exact_solution"]):
        assert torch.allclose(param, exact_solution)
