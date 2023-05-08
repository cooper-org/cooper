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

    alternating = cooper.optim.AlternatingType("PrimalDual")

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
        primal_optimizers=primal_optimizers,
        constraint_groups=cmp.constraint_groups,
        extrapolation=False,
        alternating=alternating,
    )

    cmp = cooper_test_utils.Toy2dCMP(use_ineq_constraints=True, use_constraint_surrogate=True, device=device)

    mktensor = testing_utils.mktensor(device=device)

    alternating = cooper.optim.AlternatingType("PrimalDual")
    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
        primal_optimizers=primal_optimizers,
        constraint_groups=cmp.constraint_groups,
        extrapolation=False,
        alternating=alternating,
        dual_optimizer_name="SGD",
        dual_optimizer_kwargs={"lr": 1e-2},
    )

    roll_kwargs = {
        "compute_cmp_state_fn": lambda: cmp.compute_cmp_state(params),
        "compute_violations_fn": (lambda: cmp.compute_violations(params)) if use_violation_fn else None,
        "return_multipliers": True,
    }

    x0_y0 = mktensor([0.0, -1.0])
    lmbda0 = mktensor([0.0, 0.0])

    # ----------------------- First iteration -----------------------
    # The returned LagrangianStore is computed after the primal update but before the
    # dual update.
    cmp_state, lagrangian_store_after_primal_update = cooper_optimizer.roll(**roll_kwargs)

    # No dual update yet, so the observed multipliers should be zero, matching lmdba0
    assert torch.allclose(torch.cat(lagrangian_store_after_primal_update.observed_multipliers), lmbda0)

    # analytical_gradients computes the gradients of the loss and surrogate constraints
    grads_x0_y0 = cmp.analytical_gradients(x0_y0)
    x1_y1 = x0_y0 - 1e-2 * (grads_x0_y0[0] + torch.sum(lmbda0 * grads_x0_y0[1]))
    assert torch.allclose(params, x1_y1)

    # Check primal and dual Lagrangians.
    violations_after_primal_update = mktensor([_[1].violation for _ in cmp_state.observed_constraints])
    strict_violations_after_primal_update = mktensor([_[1].strict_violation for _ in cmp_state.observed_constraints])

    if use_violation_fn:
        # The loss is not evaluated at the updated primal point, so cmp_state.loss=None
        assert cmp_state.loss is None
        loss = torch.tensor(0.0, device=device)
    else:
        loss = cmp_state.loss

    primal_lag0 = loss + torch.sum(violations_after_primal_update * lmbda0)
    dual_lag0 = torch.sum(strict_violations_after_primal_update * lmbda0)

    assert torch.allclose(lagrangian_store_after_primal_update.lagrangian, primal_lag0)
    assert torch.allclose(lagrangian_store_after_primal_update.dual_lagrangian, dual_lag0)

    # Lambda update uses the violations after the primal update. Re-comuting them
    # manually to ensure correctness.
    strict_violations = mktensor([_[1].strict_violation for _ in cmp.compute_cmp_state(x1_y1).observed_constraints])
    lmbda1 = torch.relu(lmbda0 + 1e-2 * strict_violations)

    # ----------------------- Second iteration -----------------------
    cmp_state, lagrangian_store_after_primal_update = cooper_optimizer.roll(**roll_kwargs)

    assert torch.allclose(torch.cat(lagrangian_store_after_primal_update.observed_multipliers), lmbda1)

    grads_x1_y1 = cmp.analytical_gradients(x1_y1)
    x2_y2 = x1_y1 - 1e-2 * (grads_x1_y1[0] + torch.sum(lmbda1 * grads_x1_y1[1]))

    # NOTE: this test requires a relaxed tolerance of 1e-4
    assert torch.allclose(params, x2_y2, atol=1e-4)

    # Check primal and dual Lagrangians.
    violations_after_primal_update = mktensor([_[1].violation for _ in cmp_state.observed_constraints])
    strict_violations_after_primal_update = mktensor([_[1].strict_violation for _ in cmp_state.observed_constraints])

    if use_violation_fn:
        # The loss is not evaluated at the updated primal point, so cmp_state.loss=None
        assert cmp_state.loss is None
        loss = torch.tensor(0.0, device=device)
    else:
        loss = cmp_state.loss

    primal_lag1 = loss + torch.sum(violations_after_primal_update * lmbda1)
    dual_lag1 = torch.sum(strict_violations_after_primal_update * lmbda1)

    assert torch.allclose(lagrangian_store_after_primal_update.lagrangian, primal_lag1)
    assert torch.allclose(lagrangian_store_after_primal_update.dual_lagrangian, dual_lag1)


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

    alternating = cooper.optim.AlternatingType("DualPrimal")

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
        primal_optimizers=primal_optimizers,
        constraint_groups=cmp.constraint_groups,
        extrapolation=False,
        alternating=alternating,
    )

    cmp = cooper_test_utils.Toy2dCMP(use_ineq_constraints=True, use_constraint_surrogate=True, device=device)

    mktensor = testing_utils.mktensor(device=device)

    alternating = cooper.optim.AlternatingType("DualPrimal")
    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
        primal_optimizers=primal_optimizers,
        constraint_groups=cmp.constraint_groups,
        extrapolation=False,
        alternating=alternating,
        dual_optimizer_name="SGD",
        dual_optimizer_kwargs={"lr": 1e-2},
    )

    roll_kwargs = {"compute_cmp_state_fn": lambda: cmp.compute_cmp_state(params), "return_multipliers": True}

    x0_y0 = mktensor([0.0, -1.0])
    lmbda0 = mktensor([0.0, 0.0])

    # ----------------------- First iteration -----------------------
    # The returned LagrangianStore is computed after the dual update but before the
    # dual update.
    cmp_state, lagrangian_store_after_dual_update = cooper_optimizer.roll(**roll_kwargs)
    violations_after_dual_update = mktensor([_[1].violation for _ in cmp_state.observed_constraints])
    strict_violations_after_dual_update = mktensor([_[1].strict_violation for _ in cmp_state.observed_constraints])

    # DualPrimal optimizer only computes violations once. Then, the cmp at thr previous
    # iterate should match the cmp after the dual update.
    manual_cmp_state = cmp.compute_cmp_state(x0_y0)
    manual_violations = mktensor([_[1].violation for _ in manual_cmp_state.observed_constraints])
    manual_strict_violations = mktensor([_[1].strict_violation for _ in manual_cmp_state.observed_constraints])
    assert torch.isclose(cmp_state.loss, manual_cmp_state.loss)
    assert torch.allclose(violations_after_dual_update, manual_violations)
    assert torch.allclose(strict_violations_after_dual_update, manual_strict_violations)

    # Computing the dual update manually to ensure correctness
    lmbda1 = torch.relu(lmbda0 + 1e-2 * strict_violations_after_dual_update)
    assert torch.allclose(torch.cat(lagrangian_store_after_dual_update.observed_multipliers), lmbda1)

    # Check primal and dual Lagrangians.
    primal_lag0 = cmp_state.loss + torch.sum(violations_after_dual_update * lmbda1)
    dual_lag0 = torch.sum(strict_violations_after_dual_update * lmbda1)

    assert torch.allclose(lagrangian_store_after_dual_update.lagrangian, primal_lag0)
    assert torch.allclose(lagrangian_store_after_dual_update.dual_lagrangian, dual_lag0)

    # Computing the primal update
    # analytical_gradients computes the gradients of the loss and surrogate constraints
    grads_x0_y0 = cmp.analytical_gradients(x0_y0)
    x1_y1 = x0_y0 - 1e-2 * (grads_x0_y0[0] + torch.sum(lmbda1 * grads_x0_y0[1]))
    # NOTE: this test requires a relaxed tolerance of 1e-4
    assert torch.allclose(params, x1_y1, atol=1e-4)

    # ----------------------- Second iteration -----------------------
    cmp_state, lagrangian_store_after_dual_update = cooper_optimizer.roll(**roll_kwargs)
    violations_after_dual_update = mktensor([_[1].violation for _ in cmp_state.observed_constraints])
    strict_violations_after_dual_update = mktensor([_[1].strict_violation for _ in cmp_state.observed_constraints])

    manual_cmp_state = cmp.compute_cmp_state(x1_y1)
    manual_violations = mktensor([_[1].violation for _ in manual_cmp_state.observed_constraints])
    manual_strict_violations = mktensor([_[1].strict_violation for _ in manual_cmp_state.observed_constraints])
    # NOTE: the following tests requires a relaxed tolerance of 1e-4
    assert torch.isclose(cmp_state.loss, manual_cmp_state.loss, atol=1e-4)
    assert torch.allclose(violations_after_dual_update, manual_violations, atol=1e-4)
    assert torch.allclose(strict_violations_after_dual_update, manual_strict_violations, atol=1e-4)

    lmbda2 = torch.relu(lmbda1 + 1e-2 * strict_violations_after_dual_update)
    assert torch.allclose(torch.cat(lagrangian_store_after_dual_update.observed_multipliers), lmbda2)

    primal_lag1 = cmp_state.loss + torch.sum(violations_after_dual_update * lmbda2)
    dual_lag1 = torch.sum(strict_violations_after_dual_update * lmbda2)

    assert torch.allclose(lagrangian_store_after_dual_update.lagrangian, primal_lag1)
    assert torch.allclose(lagrangian_store_after_dual_update.dual_lagrangian, dual_lag1)

    grads_x1_y1 = cmp.analytical_gradients(x1_y1)
    x2_y2 = x1_y1 - 1e-2 * (grads_x1_y1[0] + torch.sum(lmbda2 * grads_x1_y1[1]))

    # NOTE: this test requires a relaxed tolerance of 1e-4
    assert torch.allclose(params, x2_y2, atol=1e-4)


@pytest.mark.parametrize(
    "alternating_type", [cooper.optim.AlternatingType.PRIMAL_DUAL, cooper.optim.AlternatingType.DUAL_PRIMAL]
)
@pytest.mark.parametrize("use_defect_fn", [True, False])
def test_convergence_surrogate(alternating_type, use_defect_fn, Toy2dCMP_problem_properties, device):
    pass
