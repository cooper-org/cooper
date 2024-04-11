import cooper_test_utils
import pytest
import testing_utils
import torch

import cooper

USE_CONSTRAINT_SURROGATE = True


def test_manual_simultaneous_surrogate(Toy2dCMP_problem_properties, Toy2dCMP_params_init, device):
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

    # Only performing this test for the case of a single primal optimizer
    assert isinstance(params, torch.nn.Parameter)

    cmp = cooper_test_utils.Toy2dCMP(
        use_ineq_constraints=True, use_constraint_surrogate=USE_CONSTRAINT_SURROGATE, device=device
    )

    mktensor = testing_utils.mktensor(device=device)

    PRIMAL_LR = 1e-2
    DUAL_LR = 1e-2

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
        primal_optimizers=primal_optimizers,
        cmp=cmp,
        extrapolation=False,
        alternation_type=cooper.optim.AlternationType.FALSE,
        dual_optimizer_class=torch.optim.SGD,
        dual_optimizer_kwargs={"lr": DUAL_LR},
    )

    roll_kwargs = {"compute_cmp_state_kwargs": dict(params=params)}

    x0_y0 = mktensor([0.0, -1.0])
    lmbda0 = mktensor([0.0, 0.0])

    # ----------------------- First iteration -----------------------
    # The CMPState returned when using simultaneous updates is measured _before any_ updates
    loss, cmp_state, primal_ls, dual_ls = cooper_optimizer.roll(**roll_kwargs)
    _cmp_state = cmp.compute_cmp_state(x0_y0)

    strict_violations_before_primal_update = mktensor(list(cmp_state.observed_strict_violations()))
    lmbda1 = torch.relu(lmbda0 + DUAL_LR * strict_violations_before_primal_update)

    # The primal Lagrangian store uses the original multipliers
    observed_multipliers = torch.cat(list(primal_ls.observed_multiplier_values()))
    assert torch.allclose(observed_multipliers, lmbda0)

    # analytical_gradients computes the gradients of the loss and surrogate constraints
    grads_x0_y0 = cmp.analytical_gradients(x0_y0)
    # Need to use the original multipliers when computing the primal update
    x1_y1 = x0_y0 - PRIMAL_LR * (grads_x0_y0[0] + torch.sum(lmbda0 * grads_x0_y0[1]))
    # TODO(galllego-posada): check if need for relaxed tolerance comes from using surrogate constraints?
    assert torch.allclose(params, x1_y1, atol=1e-4)

    # The value of the dual Lagrangian should only count the constraint contributions
    # before the primal update, using the original multipliers
    assert torch.allclose(dual_ls.lagrangian, torch.sum(strict_violations_before_primal_update * lmbda0))

    # The value of the primal Lagrangian should be the loss at (x0, y0) plus the
    # constraint contributions; using both original multiplier and primal params
    violations_before_primal_update = mktensor(list(_cmp_state.observed_violations()))
    assert torch.allclose(primal_ls.lagrangian, _cmp_state.loss + torch.sum(violations_before_primal_update * lmbda0))

    # ----------------------- Second iteration -----------------------
    loss, cmp_state, primal_ls, dual_ls = cooper_optimizer.roll(**roll_kwargs)
    _cmp_state = cmp.compute_cmp_state(x1_y1)

    strict_violations_before_primal_update = mktensor(list(cmp_state.observed_strict_violations()))
    lmbda2 = torch.relu(lmbda1 + DUAL_LR * strict_violations_before_primal_update)  # noqa: F841

    # The primal Lagrangian store uses the multipliers before updating them
    observed_multipliers = torch.cat(list(primal_ls.observed_multiplier_values()))
    assert torch.allclose(observed_multipliers, lmbda1)

    # analytical_gradients computes the gradients of the loss and surrogate constraints
    grads_x1_y1 = cmp.analytical_gradients(x1_y1)
    # Need to use the received multipliers when computing the primal update
    x2_y2 = x1_y1 - PRIMAL_LR * (grads_x1_y1[0] + torch.sum(lmbda1 * grads_x1_y1[1]))
    # TODO(galllego-posada): check if need for relaxed tolerance comes from using surrogate constraints?
    assert torch.allclose(params, x2_y2, atol=1e-4)

    # The value of the dual Lagrangian should only count the constraint contributions
    # before the second primal update, using multipliers before their second update
    assert torch.allclose(dual_ls.lagrangian, torch.sum(strict_violations_before_primal_update * lmbda1))

    # The value of the primal Lagrangian should be the loss at (x1, y1) plus the
    # constraint contributions _before_ the second primal or dual updates
    violations_before_primal_update = mktensor(list(_cmp_state.observed_violations()))
    target_primal_lagrangian = _cmp_state.loss + torch.sum(violations_before_primal_update * lmbda1)
    # TODO(galllego-posada): check if need for relaxed tolerance comes from using surrogate constraints?
    assert torch.allclose(primal_ls.lagrangian, target_primal_lagrangian, atol=1e-4)


# TODO(gallego-posada): identify what the blocker is for implementing this test and report back
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

#     cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(primal_optimizers, cmp=cmp)

#     for step_id in range(3000):
#         compute_cmp_state_fn = lambda: cmp.compute_cmp_state(params)
#         cmp_state, lagrangian_store = cooper_optimizer.roll(compute_cmp_state_fn=compute_cmp_state_fn)

#     for param, exact_solution in zip(params, Toy2dCMP_problem_properties["exact_solution"]):
#         assert torch.allclose(param, exact_solution)
