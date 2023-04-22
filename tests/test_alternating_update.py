import cooper_test_utils
import pytest
import testing_utils
import torch


@pytest.mark.parametrize("alternating", [True, False])
@pytest.mark.parametrize("use_defect_fn", [True, False])
def test_manual_alternating(alternating, use_defect_fn, Toy2dCMP_problem_properties, Toy2dCMP_params_init, device):
    """Test first step of alternating GDA updates on toy 2D problem."""

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
        constraint_groups=cmp.constraint_groups,
        extrapolation=False,
        alternating=alternating,
    )

    if use_defect_fn:
        compute_cmp_state_fn = None
        compute_violations_fn = lambda: cmp.compute_violations(params, existing_cmp_state=None)
    else:
        compute_cmp_state_fn = lambda: cmp.compute_cmp_state(params)
        compute_violations_fn = None

    cooper_optimizer.zero_grad()
    cmp_state = cmp.compute_cmp_state(params)
    lagrangian_store = cmp_state.populate_lagrangian(return_multipliers=True)
    lagrangian, observed_multipliers = lagrangian_store.lagrangian, lagrangian_store.observed_multipliers

    # Check loss, proxy and non-proxy defects after forward pass
    assert torch.allclose(lagrangian, mktensor(2.0))
    assert torch.allclose(cmp_state.loss, mktensor(2.0))
    for (constraint_group, constraint_state), target_value in zip(cmp_state.observed_constraints, [2.0, -2.0]):
        assert torch.allclose(constraint_state.violation, mktensor([target_value]))
    # Multiplier initialization
    for multiplier, target_value in zip(observed_multipliers, [0.0, 0.0]):
        assert torch.allclose(multiplier, mktensor([target_value]))

    cmp_state.backward()
    # Check primal and dual gradients after backward. Dual gradient must match the
    # constraint violations.
    assert torch.allclose(params.grad, mktensor([0.0, -4.0]))
    for multiplier, (constraint_group, constraint_state) in zip(observed_multipliers, cmp_state.observed_constraints):
        assert torch.allclose(multiplier.grad, constraint_state.violation)

    # Check updated primal and dual variable values
    if alternating:
        cmp_state_after_primal_update, multipliers_after_alternating_update = cooper_optimizer.step(
            compute_cmp_state_fn=compute_cmp_state_fn,
            compute_violations_fn=compute_violations_fn,
            return_multipliers=True,
        )
    else:
        cooper_optimizer.step()

    # After performing the update to the primal variables, their value is [0, -0.96].
    assert torch.allclose(params, mktensor([0.0, -0.96]))

    if alternating:
        if use_defect_fn:
            # If we use defect_fn, the loss at this point may not have been recomputed
            pass
        else:
            # If alternating step uses the compute_cmp_state_fn, the loss is computed at
            # the new value of the primal variables ([0, -0.96]).
            assert torch.allclose(cmp_state_after_primal_update.loss, mktensor([1.8432]))

        # The constraint violations after the primal update are are [1.96, -1.96], so the
        # value of the multipliers is the violations times the dual LR (1e-2). Note that
        # the second constraint in feasible, so the second multiplier remains at 0.
        for multiplier, target_value in zip(multipliers_after_alternating_update, [1.96 * 1e-2, 0.0]):
            assert torch.allclose(multiplier, mktensor([target_value]))

    else:
        # Loss and violation measurements taken the initialization point [0, -1]
        assert torch.allclose(cmp_state.loss, mktensor([2.0]))

        # In this case the constraint violations are [2., -2.]
        multipliers_after_simultaneous_update = [
            constraint_group.multiplier() for constraint_group, constraint_state in cmp_state.observed_constraints
        ]
        for multiplier, target_value in zip(multipliers_after_simultaneous_update, [2.0 * 1e-2, 0.0]):
            assert torch.allclose(multiplier, mktensor([target_value]))


@pytest.mark.parametrize("alternating", [True, False])
@pytest.mark.parametrize("use_defect_fn", [True, False])
def test_convergence_alternating(
    alternating,
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
        primal_optimizers=primal_optimizers,
        constraint_groups=cmp.constraint_groups,
        extrapolation=False,
        alternating=True,
    )

    if use_defect_fn:
        compute_cmp_state_fn = None
        compute_violations_fn = lambda: cmp.compute_violations(params, existing_cmp_state=None)
    else:
        compute_cmp_state_fn = lambda: cmp.compute_cmp_state(params)
        compute_violations_fn = None

    for step_id in range(1500):
        cooper_optimizer.zero_grad()
        cmp_state = cmp.compute_cmp_state(params)
        _ = cmp_state.populate_lagrangian(return_multipliers=True)
        cmp_state.backward()
        _ = cooper_optimizer.step(
            compute_cmp_state_fn=compute_cmp_state_fn, compute_violations_fn=compute_violations_fn
        )

    for param, exact_solution in zip(params, Toy2dCMP_problem_properties["exact_solution"]):
        assert torch.allclose(param, exact_solution)
