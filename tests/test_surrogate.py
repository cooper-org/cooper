#!/usr/bin/env python

"""Tests for Constrained Optimizer class."""

import cooper_test_utils
import pytest
import testing_utils
import torch

import cooper


def test_manual_proxy(Toy2dCMP_problem_properties, Toy2dCMP_params_init, device):
    """Test first step of simultaneous GDA updates with surrogates on toy 2D problem."""

    if not Toy2dCMP_problem_properties["use_ineq_constraints"]:
        pytest.skip("Surrogate update tests require a problem with constraints.")

    if not torch.allclose(Toy2dCMP_params_init, torch.tensor([0.0, -1.0], device=device)):
        pytest.skip("Manual surrogate test only considers the case of initialization at [0, -1]")

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=False, params_init=Toy2dCMP_params_init
    )

    # Only perfoming this test for the case of a single primal optimizer
    assert isinstance(params, torch.nn.Parameter)

    cmp = cooper_test_utils.Toy2dCMP(use_ineq_constraints=True, use_constraint_surrogate=True, device=device)

    mktensor = testing_utils.mktensor(device=device)

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
        primal_optimizers=primal_optimizers,
        multipliers=cmp.multipliers,
        extrapolation=False,
        alternation_type=cooper.optim.AlternationType.FALSE,
        dual_optimizer_class=torch.optim.SGD,
        dual_optimizer_kwargs={"lr": 1e-2},
    )

    roll_kwargs = {"compute_cmp_state_fn": lambda: cmp.compute_cmp_state(params)}

    x0_y0 = mktensor([0.0, -1.0])
    lmbda0 = mktensor([0.0, 0.0])

    # ------------ First step of surrogate updates ------------
    cmp_state, lagrangian_store = cooper_optimizer.roll(**roll_kwargs)
    violations = mktensor([_[1].violation for _ in cmp_state.observed_constraints])
    strict_violations = mktensor([_[1].strict_violation for _ in cmp_state.observed_constraints])

    # Check primal and dual Lagrangians
    primal_lagrangian0 = cmp_state.loss + torch.sum(violations * lmbda0)
    dual_lagrangian0 = torch.sum(strict_violations * lmbda0)

    assert torch.allclose(lagrangian_store.lagrangian, primal_lagrangian0)
    assert torch.allclose(lagrangian_store.dual_lagrangian, dual_lagrangian0)

    # analytical_gradients computes the gradients of the loss and surrogate constraints
    grads_x0_y0 = cmp.analytical_gradients(x0_y0)
    x1_y1 = x0_y0 - 1e-2 * (grads_x0_y0[0] + torch.sum(lmbda0 * grads_x0_y0[1]))
    assert torch.allclose(params, x1_y1)

    # Observed multipliers should be zero, matching lmdba0
    assert torch.allclose(torch.cat(lagrangian_store.multiplier_values_for_primal_constraints()), lmbda0)

    lmbda1 = torch.relu(lmbda0 + 1e-2 * strict_violations)

    # ------------ Second step of surrogate updates ------------
    cmp_state, lagrangian_store = cooper_optimizer.roll(**roll_kwargs)
    violations = mktensor([_[1].violation for _ in cmp_state.observed_constraints])
    strict_violations = mktensor([_[1].strict_violation for _ in cmp_state.observed_constraints])

    # Check primal and dual Lagrangians
    primal_lagrangian1 = cmp_state.loss + torch.sum(violations * lmbda1)
    dual_lagrangian1 = torch.sum(strict_violations * lmbda1)

    assert torch.allclose(lagrangian_store.lagrangian, primal_lagrangian1)
    assert torch.allclose(lagrangian_store.dual_lagrangian, dual_lagrangian1)

    # analytical_gradients computes the gradients of the loss and surrogate constraints
    grads_x1_y1 = cmp.analytical_gradients(x1_y1)
    x2_y2 = x1_y1 - 1e-2 * (grads_x1_y1[0] + torch.sum(lmbda1 * grads_x1_y1[1]))

    # NOTE: this test requires a relaxed tolerance of 1e-4
    assert torch.allclose(params, x2_y2, atol=1e-4)

    assert torch.allclose(torch.cat(lagrangian_store.multiplier_values_for_primal_constraints()), lmbda1)


# TODO(juan43ramirez): implement
# def test_convergence_surrogate(
#     Toy2dCMP_problem_properties, Toy2dCMP_params_init, use_multiple_primal_optimizers, device
# ):
#     params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
#         use_multiple_primal_optimizers, Toy2dCMP_params_init
#     )

#     use_ineq_constraints = Toy2dCMP_problem_properties["use_ineq_constraints"]
#     if not use_ineq_constraints:
#         pytest.skip("Surrogate update tests require a problem with constraints.")
#     use_constraint_surrogate = True

#     cmp = cooper_test_utils.Toy2dCMP(
#         use_ineq_constraints=use_ineq_constraints, use_constraint_surrogate=use_constraint_surrogate, device=device
#     )

#     cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(primal_optimizers, multipliers=cmp.multipliers)

#     for step_id in range(3000):
#         compute_cmp_state_fn = lambda: cmp.compute_cmp_state(params)
#         cmp_state, lagrangian_store = cooper_optimizer.roll(compute_cmp_state_fn=compute_cmp_state_fn)

#     for param, exact_solution in zip(params, Toy2dCMP_problem_properties["exact_solution"]):
#         assert torch.allclose(param, exact_solution)
