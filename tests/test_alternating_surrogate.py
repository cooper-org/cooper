#!/usr/bin/env python

"""Tests for Constrained Optimizer class."""

import cooper_test_utils
import pytest
import testing_utils
import torch

import cooper


@pytest.mark.parametrize("use_violation_fn", [True, False])
def test_manual_PrimalDual_surrogate(use_violation_fn, Toy2dCMP_problem_properties, Toy2dCMP_params_init, device):
    """Test first two iterations of PrimalDual alternating GDA updates on a toy 2D
    problem with surrogate constraints."""

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
        multipliers=cmp.multipliers,
        extrapolation=False,
        alternation_type=cooper.optim.AlternationType.PRIMAL_DUAL,
    )

    cmp = cooper_test_utils.Toy2dCMP(use_ineq_constraints=True, use_constraint_surrogate=True, device=device)

    mktensor = testing_utils.mktensor(device=device)

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
        primal_optimizers=primal_optimizers,
        multipliers=cmp.multipliers,
        extrapolation=False,
        alternation_type=cooper.optim.AlternationType.PRIMAL_DUAL,
        dual_optimizer_name="SGD",
        dual_optimizer_kwargs={"lr": 1e-2},
    )

    roll_kwargs = {
        "compute_cmp_state_fn": lambda: cmp.compute_cmp_state(params),
        "compute_violations_fn": (lambda: cmp.compute_violations(params)) if use_violation_fn else None,
    }

    x0_y0 = mktensor([0.0, -1.0])
    lmbda0 = mktensor([0.0, 0.0])

    # ----------------------- First iteration -----------------------
    # The returned LagrangianStore is computed after the primal update but before the
    # dual update.
    cmp_state, lagrangian_store_after_primal_update = cooper_optimizer.roll(**roll_kwargs)

    # No dual update yet, so the observed multipliers should be zero, matching lmdba0
    observed_multipliers = torch.cat(lagrangian_store_after_primal_update.multiplier_values_for_primal_constraints())
    assert torch.allclose(observed_multipliers, lmbda0)

    # analytical_gradients computes the gradients of the loss and surrogate constraints
    grads_x0_y0 = cmp.analytical_gradients(x0_y0)
    x1_y1 = x0_y0 - 1e-2 * (grads_x0_y0[0] + torch.sum(lmbda0 * grads_x0_y0[1]))
    assert torch.allclose(params, x1_y1)

    # Check primal and dual Lagrangians.
    violations_after_primal_update = mktensor([_[1].violation for _ in cmp_state.observed_constraints])
    strict_violations_after_primal_update = mktensor([_[1].strict_violation for _ in cmp_state.observed_constraints])

    loss = cmp_state.loss
    primal_lagrangian0 = loss + torch.sum(violations_after_primal_update * lmbda0)
    dual_lagrangian0 = torch.sum(strict_violations_after_primal_update * lmbda0)

    assert torch.allclose(lagrangian_store_after_primal_update.lagrangian, primal_lagrangian0)
    assert torch.allclose(lagrangian_store_after_primal_update.dual_lagrangian, dual_lagrangian0)

    # Lambda update uses the violations after the primal update. Re-comuting them
    # manually to ensure correctness.
    strict_violations = mktensor([_[1].strict_violation for _ in cmp.compute_cmp_state(x1_y1).observed_constraints])
    lmbda1 = torch.relu(lmbda0 + 1e-2 * strict_violations)

    # ----------------------- Second iteration -----------------------
    cmp_state, lagrangian_store_after_primal_update = cooper_optimizer.roll(**roll_kwargs)

    observed_multipliers = torch.cat(lagrangian_store_after_primal_update.multiplier_values_for_primal_constraints())
    assert torch.allclose(observed_multipliers, lmbda1)

    grads_x1_y1 = cmp.analytical_gradients(x1_y1)
    x2_y2 = x1_y1 - 1e-2 * (grads_x1_y1[0] + torch.sum(lmbda1 * grads_x1_y1[1]))

    # NOTE: this test requires a relaxed tolerance of 1e-4
    assert torch.allclose(params, x2_y2, atol=1e-4)

    # Check primal and dual Lagrangians.
    violations_after_primal_update = mktensor([_[1].violation for _ in cmp_state.observed_constraints])
    strict_violations_after_primal_update = mktensor([_[1].strict_violation for _ in cmp_state.observed_constraints])

    loss = cmp_state.loss
    primal_lagrangian1 = loss + torch.sum(violations_after_primal_update * lmbda1)
    dual_lagrangian1 = torch.sum(strict_violations_after_primal_update * lmbda1)

    assert torch.allclose(lagrangian_store_after_primal_update.lagrangian, primal_lagrangian1)
    assert torch.allclose(lagrangian_store_after_primal_update.dual_lagrangian, dual_lagrangian1)


def test_manual_DualPrimal_surrogate(Toy2dCMP_problem_properties, Toy2dCMP_params_init, device):
    """Test first two iterations of DualPrimal alternating GDA updates on a toy 2D
    problem with surrogate constraints."""

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
        multipliers=cmp.multipliers,
        extrapolation=False,
        alternation_type=cooper.optim.AlternationType.DUAL_PRIMAL,
    )

    cmp = cooper_test_utils.Toy2dCMP(use_ineq_constraints=True, use_constraint_surrogate=True, device=device)

    mktensor = testing_utils.mktensor(device=device)

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
        primal_optimizers=primal_optimizers,
        multipliers=cmp.multipliers,
        extrapolation=False,
        alternation_type=cooper.optim.AlternationType.DUAL_PRIMAL,
        dual_optimizer_name="SGD",
        dual_optimizer_kwargs={"lr": 1e-2},
    )

    roll_kwargs = {"compute_cmp_state_fn": lambda: cmp.compute_cmp_state(params)}

    x0_y0 = mktensor([0.0, -1.0])
    lmbda0 = mktensor([0.0, 0.0])

    # ----------------------- First iteration -----------------------

    cmp_state, lagrangian_store_after_roll = cooper_optimizer.roll(**roll_kwargs)
    violations = mktensor([_[1].violation for _ in cmp_state.observed_constraints])
    strict_violations = mktensor([_[1].strict_violation for _ in cmp_state.observed_constraints])

    # DualPrimal optimizer only computes the CMPSTate and violations once, using the
    # existing primal parameters. Therefore, the CMPState using the iterates _before_
    # the roll should match the CMPState returned by the roll.
    manual_cmp_state = cmp.compute_cmp_state(x0_y0)
    manual_violations = mktensor([_[1].violation for _ in manual_cmp_state.observed_constraints])
    manual_strict_violations = mktensor([_[1].strict_violation for _ in manual_cmp_state.observed_constraints])
    assert torch.isclose(cmp_state.loss, manual_cmp_state.loss)
    assert torch.allclose(violations, manual_violations)
    assert torch.allclose(strict_violations, manual_strict_violations)

    # Computing the dual update manually to ensure correctness
    # The multipliers are updated based on the strict violations
    lmbda1 = torch.relu(lmbda0 + 1e-2 * strict_violations)
    # After the dual update, the multipliers will be used to weight the (potentially
    # differentiable) violations in the primal Lagrangian
    observed_multipliers = torch.cat(lagrangian_store_after_roll.multiplier_values_for_primal_constraints())
    assert torch.allclose(observed_multipliers, lmbda1)
    primal_lagrangian0 = cmp_state.loss + torch.sum(violations * lmbda1)
    assert torch.allclose(lagrangian_store_after_roll.lagrangian, primal_lagrangian0)

    # The dual Lagrangian returned by the LagrangianStore is computed using the dual
    # variables _before_ the dual update, so it should match the Lagrangian computed
    # using lmbda0 (the dual parameters before roll)
    dual_lagrangian0 = torch.sum(strict_violations * lmbda0)
    if not torch.allclose(lagrangian_store_after_roll.dual_lagrangian, dual_lagrangian0):
        breakpoint()

    # Computing the primal update
    # analytical_gradients computes the gradients of the loss and surrogate constraints
    grads_x0_y0 = cmp.analytical_gradients(x0_y0)
    x1_y1 = x0_y0 - 1e-2 * (grads_x0_y0[0] + torch.sum(lmbda1 * grads_x0_y0[1]))
    # NOTE: this test requires a relaxed tolerance of 1e-4
    assert torch.allclose(params, x1_y1, atol=1e-4)

    # ----------------------- Second iteration -----------------------
    cmp_state, lagrangian_store_after_roll = cooper_optimizer.roll(**roll_kwargs)
    violations = mktensor([_[1].violation for _ in cmp_state.observed_constraints])
    strict_violations = mktensor([_[1].strict_violation for _ in cmp_state.observed_constraints])

    manual_cmp_state = cmp.compute_cmp_state(x1_y1)
    manual_violations = mktensor([_[1].violation for _ in manual_cmp_state.observed_constraints])
    manual_strict_violations = mktensor([_[1].strict_violation for _ in manual_cmp_state.observed_constraints])
    # NOTE: the following tests requires a relaxed tolerance of 1e-4
    assert torch.isclose(cmp_state.loss, manual_cmp_state.loss, atol=1e-4)
    assert torch.allclose(violations, manual_violations, atol=1e-4)
    assert torch.allclose(strict_violations, manual_strict_violations, atol=1e-4)

    lmbda2 = torch.relu(lmbda1 + 1e-2 * strict_violations)

    observed_multipliers = torch.cat(lagrangian_store_after_roll.multiplier_values_for_primal_constraints())
    assert torch.allclose(observed_multipliers, lmbda2)

    primal_lagrangian1 = cmp_state.loss + torch.sum(violations * lmbda2)
    assert torch.allclose(lagrangian_store_after_roll.lagrangian, primal_lagrangian1)

    dual_lagrangian1 = torch.sum(strict_violations * lmbda1)
    assert torch.allclose(lagrangian_store_after_roll.dual_lagrangian, dual_lagrangian1)

    grads_x1_y1 = cmp.analytical_gradients(x1_y1)
    x2_y2 = x1_y1 - 1e-2 * (grads_x1_y1[0] + torch.sum(lmbda2 * grads_x1_y1[1]))

    # NOTE: this test requires a relaxed tolerance of 1e-4
    assert torch.allclose(params, x2_y2, atol=1e-4)


# TODO(juan43ramirez): implement


@pytest.mark.parametrize(
    "alternation_type", [cooper.optim.AlternationType.PRIMAL_DUAL, cooper.optim.AlternationType.DUAL_PRIMAL]
)
@pytest.mark.parametrize("use_defect_fn", [True, False])
def test_convergence_surrogate(alternation_type, use_defect_fn, Toy2dCMP_problem_properties, device):
    pass
