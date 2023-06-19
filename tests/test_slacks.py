# import cooper_test_utils
# import pytest
# import testing_utils
# import torch

# import cooper


# def test_manual_with_slacks(use_violation_fn, Toy2dCMP_problem_properties, Toy2dCMP_params_init, device):

#     use_ineq_constraints = Toy2dCMP_problem_properties["use_ineq_constraints"]
#     if not use_ineq_constraints:
#         pytest.skip("Alternating updates requires a problem with constraints.")

#     if not torch.allclose(Toy2dCMP_params_init, torch.tensor([0.0, -1.0], device=device)):
#         pytest.skip("Manual alternating test only considers the case of initialization at [0, -1]")

#     params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
#         use_multiple_primal_optimizers=False, params_init=Toy2dCMP_params_init
#     )

#     # Only perfoming this test for the case of a single primal optimizer
#     assert isinstance(params, torch.nn.Parameter)

#     # Initializing the slack variables to zero
#     slack_variables = [torch.tensor([0.0], device=device, requires_grad=True) for _ in range(2)]
#     slack_optimizer = torch.optim.SGD(slack_variables, lr=1e-2)

#     cmp = cooper_test_utils.Toy2dCMP(use_ineq_constraints=use_ineq_constraints, slack_variables=slack_variables, device=device)

#     mktensor = testing_utils.mktensor(device=device)

#     alternating = cooper.optim.AlternatingType("PrimalDual")

#     cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
#         primal_optimizers=primal_optimizers,
#         constraint_groups=cmp.constraint_groups,
#         extrapolation=False,
#         alternating=alternating,
#         dual_optimizer_name="SGD",
#         dual_optimizer_kwargs={"lr": 1e-2},
#     )

#     roll_kwargs = {
#         "compute_cmp_state_fn": lambda: cmp.compute_cmp_state(params),
#         "compute_violations_fn": (lambda: cmp.compute_violations(params)) if use_violation_fn else None,
#         "return_multipliers": True,
#     }

#     x0_y0 = mktensor([0.0, -1.0])
#     lmbda0 = mktensor([0.0, 0.0])

#     # ------------ First step of alternating updates ------------
#     _cmp_state, lagrangian_store = cooper_optimizer.roll(**roll_kwargs)

#     x1_y1 = mktensor([0.0, -0.96])
#     assert torch.allclose(params, x1_y1)

#     cmp_state = cmp.compute_cmp_state(x1_y1)

#     violations = mktensor([_[1].violation for _ in cmp_state.observed_constraints])
#     lmbda1 = torch.relu(lmbda0 + 1e-2 * violations)

#     multiplier_values = [constraint.multiplier() for constraint, _ in cmp_state.observed_constraints]
#     for multiplier_value, target_value in zip(multiplier_values, lmbda1):
#         assert torch.allclose(multiplier_value, mktensor([target_value]))

#     if use_violation_fn:
#         # We don't see the value of the loss at the updated point since we only
#         # evaluate the violations
#         lag1 = torch.sum(violations * lmbda0)
#         # When the final Lagrangian is evaluated, the primal variables have changed,
#         # but the multipliers are still zero (not yet updated)
#         assert torch.allclose(lagrangian_store.lagrangian, lag1)
#     else:
#         lag1 = cmp_state.loss + torch.sum(violations * lmbda0)
#         # Since the multipliers are still zero, the Lagrangian matches the loss at
#         # the updated primal point
#         assert torch.allclose(lagrangian_store.lagrangian, lag1)

#     # ------------ Second step of alternating updates ------------
#     _cmp_state, lagrangian_store = cooper_optimizer.roll(**roll_kwargs)

#     x2_y2 = mktensor([0.0196 * 0.01, -0.96 + 3.8596 * 1e-2])
#     assert torch.allclose(params, x2_y2)

#     cmp_state = cmp.compute_cmp_state(x2_y2)

#     violations = mktensor([_[1].violation for _ in cmp_state.observed_constraints])
#     lmbda2 = torch.relu(lmbda1 + 1e-2 * violations)

#     multiplier_values = [constraint.multiplier() for constraint, _ in cmp_state.observed_constraints]
#     for multiplier, target_value in zip(multiplier_values, lmbda2):
#         assert torch.allclose(multiplier, mktensor([target_value]))

#     if use_violation_fn:
#         # We don't see the value of the loss at the updated point since we only
#         # evaluate the violations
#         lag2 = torch.sum(violations * lmbda1)
#         # When the final Lagrangian is evaluated, the primal variables have changed,
#         # but the multipliers are still zero (not yet updated)
#         assert torch.allclose(lagrangian_store.lagrangian, lag2)
#     else:
#         lag2 = cmp_state.loss + torch.sum(violations * lmbda1)
#         # Since the multipliers are still zero, the Lagrangian matches the loss at
#         # the updated primal point
#         assert torch.allclose(lagrangian_store.lagrangian, lag2)

# @pytest.mark.parametrize(
#     "alternating_type", [cooper.optim.AlternatingType.PRIMAL_DUAL, cooper.optim.AlternatingType.DUAL_PRIMAL]
# )
# @pytest.mark.parametrize("use_defect_fn", [True, False])
# def test_convergence_alternating(
#     alternating_type,
#     use_defect_fn,
#     Toy2dCMP_problem_properties,
#     Toy2dCMP_params_init,
#     use_multiple_primal_optimizers,
#     device,
# ):
#     """Test convergence of alternating GDA updates on toy 2D problem."""

#     use_ineq_constraints = Toy2dCMP_problem_properties["use_ineq_constraints"]
#     if not use_ineq_constraints:
#         pytest.skip("Alternating updates requires a problem with constraints.")

#     params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
#         use_multiple_primal_optimizers=use_multiple_primal_optimizers, params_init=Toy2dCMP_params_init
#     )

#     cmp = cooper_test_utils.Toy2dCMP(use_ineq_constraints=use_ineq_constraints, device=device)

#     cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
#         primal_optimizers=primal_optimizers,
#         constraint_groups=cmp.constraint_groups,
#         extrapolation=False,
#         alternating=alternating_type,
#     )

#     compute_cmp_state_fn = lambda: cmp.compute_cmp_state(params)
#     compute_violations_fn = (lambda: cmp.compute_violations(params)) if use_defect_fn else None

#     roll_kwargs = {"compute_cmp_state_fn": compute_cmp_state_fn}
#     if alternating_type == cooper.optim.AlternatingType.PRIMAL_DUAL:
#         roll_kwargs["compute_violations_fn"] = compute_violations_fn

#     for step_id in range(1500):
#         cooper_optimizer.roll(**roll_kwargs)

#     for param, exact_solution in zip(params, Toy2dCMP_problem_properties["exact_solution"]):
#         assert torch.allclose(param, exact_solution)
