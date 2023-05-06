#!/usr/bin/env python

"""Tests for Constrained Optimizer class."""

import cooper_test_utils
import pytest
import testing_utils
import torch

import cooper


def test_manual_alternating_surrogate(Toy2dCMP_problem_properties, Toy2dCMP_params_init, device):
    """Test first two iterations of alternating GDA updates on a toy 2D problem with
    surrogate constraints."""

    use_ineq_constraints = Toy2dCMP_problem_properties["use_ineq_constraints"]
    if not use_ineq_constraints:
        pytest.skip("Alternating updates requires a problem with constraints.")

    if not torch.allclose(Toy2dCMP_params_init, torch.tensor([0.0, -1.0], device=device)):
        pytest.skip("Manual alternating test only considers the case of initialization at [0, -1]")

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=False,
        params_init=Toy2dCMP_params_init,
        optimizer_names="SGD",
        optimizer_kwargs={"lr": 5e-2},
    )

    # Only perfoming this test for the case of a single primal optimizer
    assert isinstance(params, torch.nn.Parameter)

    cmp = cooper_test_utils.Toy2dCMP(use_ineq_constraints=True, use_constraint_surrogate=True, device=device)

    mktensor = testing_utils.mktensor(device=device)

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
        primal_optimizers=primal_optimizers,
        constraint_groups=cmp.constraint_groups,
        extrapolation=False,
        alternating=cooper.optim.AlternatingType.DUAL_PRIMAL,
        dual_optimizer_name="SGD",
        dual_optimizer_kwargs={"lr": 1e-2},
    )

    # ----------------------- First iteration -----------------------
    cooper_optimizer.zero_grad()
    cmp_state = cmp.compute_cmp_state(params)
    lagrangian_store = cmp_state.populate_lagrangian(return_multipliers=True)
    lagrangian, observed_multipliers = lagrangian_store.lagrangian, lagrangian_store.observed_multipliers

    # Check loss, proxy and non-proxy defects after forward pass
    assert torch.allclose(lagrangian, mktensor(2.0))
    assert torch.allclose(cmp_state.loss, mktensor(2.0))

    for (constraint_group, constraint_state), target_value in zip(cmp_state.observed_constraints, [2.0, -1.9]):
        assert torch.allclose(constraint_state.violation, mktensor([target_value]))
        assert constraint_state.violation.requires_grad

    for (constraint_group, constraint_state), target_value in zip(cmp_state.observed_constraints, [2.0, -2.0]):
        assert torch.allclose(constraint_state.strict_violation, mktensor([target_value]))

    # Multiplier initialization
    for multiplier, target_value in zip(observed_multipliers, [0.0, 0.0]):
        assert torch.allclose(multiplier, mktensor([target_value]))

    # Check primal and dual gradients after backward. Dual gradient must match the
    # constraint (strict!) violations.
    cmp_state.backward()
    assert torch.allclose(params.grad, mktensor([0.0, -4.0]))
    for multiplier, (constraint_group, constraint_state) in zip(observed_multipliers, cmp_state.observed_constraints):
        assert torch.allclose(multiplier.weight.grad, constraint_state.strict_violation)

    # Perform alternating update
    cmp_state_after_primal_update, multipliers_after_alternating_update = cooper_optimizer.step(
        compute_cmp_state_fn=lambda: cmp.compute_cmp_state(params),
        return_multipliers=True,
    )

    assert torch.allclose(params, mktensor([0.0, -0.8]))
    # Loss and constraints are evaluated at updated primal variables
    assert torch.allclose(cmp_state_after_primal_update.loss, mktensor([2 * (-0.8) ** 2]))
    # Constraint violations [1.8, -1.8] --> this is used to update multipliers
    # Surrogate violations defects [1.8, -1.72] --> used to compute primal gradient
    for multiplier, target_value in zip(multipliers_after_alternating_update, [1.8 * 1e-2, 0.0]):
        assert torch.allclose(multiplier, mktensor([target_value]))

    # ----------------------- Second iteration -----------------------
    cooper_optimizer.zero_grad()
    cmp_state = cmp.compute_cmp_state(params)
    lagrangian_store = cmp_state.populate_lagrangian(return_multipliers=True)
    lagrangian, observed_multipliers = lagrangian_store.lagrangian, lagrangian_store.observed_multipliers

    # Check loss, proxy and non-proxy defects after forward pass
    assert torch.allclose(lagrangian, mktensor(1.3124))
    assert torch.allclose(cmp_state.loss, mktensor(1.28))

    for (constraint_group, constraint_state), target_value in zip(cmp_state.observed_constraints, [1.8, -1.72]):
        assert torch.allclose(constraint_state.violation, mktensor([target_value]))
        assert constraint_state.violation.requires_grad

    for (constraint_group, constraint_state), target_value in zip(cmp_state.observed_constraints, [1.8, -1.8]):
        assert torch.allclose(constraint_state.strict_violation, mktensor([target_value]))

    # Check primal and dual gradients after backward. Dual gradient must match the
    # constraint (strict!) violations.
    cmp_state.backward()
    assert torch.allclose(params.grad, mktensor([-0.0162, -3.218]))
    for multiplier, (constraint_group, constraint_state) in zip(observed_multipliers, cmp_state.observed_constraints):
        assert torch.allclose(multiplier.weight.grad, constraint_state.strict_violation)

    # Perform alternating update
    cmp_state_after_primal_update, multipliers_after_alternating_update = cooper_optimizer.step(
        compute_cmp_state_fn=lambda: cmp.compute_cmp_state(params),
        return_multipliers=True,
    )

    assert torch.allclose(params, mktensor([8.1e-4, -0.6391]))
    # Constraint violation at this point [1.6383, -1.63909936]
    for multiplier, target_value in zip(multipliers_after_alternating_update, [0.034383, 0.0]):
        assert torch.allclose(multiplier, mktensor([target_value]))
