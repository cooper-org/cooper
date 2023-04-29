from typing import List

import cooper_test_utils
import pytest
import testing_utils
import torch

import cooper


def test_pipeline_with_cmp(Toy2dCMP_problem_properties, Toy2dCMP_params_init, use_multiple_primal_optimizers, device):
    """Tests the correct instantiation of a simple CMP class for a 2-dimensional
    convex (quadratic) constrained problem.

    Verifies that executing simultaneous updates on this problem converges to the
    analytical solution to this problem. This check is also performed for the case where
    the CMP has no constraints -- note that the solution to the unconstrained problem is
    different.
    """

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers, Toy2dCMP_params_init
    )

    use_ineq_constraints = Toy2dCMP_problem_properties["use_ineq_constraints"]
    cmp = cooper_test_utils.Toy2dCMP(use_ineq_constraints=use_ineq_constraints, device=device)

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(primal_optimizers, cmp.constraint_groups)

    for step_id in range(1500):
        compute_cmp_state_fn = lambda: cmp.compute_cmp_state(params)
        cmp_state, lagrangian_store = cooper_optimizer.roll(compute_cmp_state_fn=compute_cmp_state_fn)

    for param, exact_solution in zip(params, Toy2dCMP_problem_properties["exact_solution"]):
        assert torch.allclose(param, exact_solution)


def test_pipeline_without_cmp(
    Toy2dCMP_problem_properties, Toy2dCMP_params_init, use_multiple_primal_optimizers, device
):
    """Test correct behavior of simultaneous updates on a 2-dimensional constrained
    problem without requiring the user to implement a CMP class explicitly. The only
    required methods are a function to evaluate the loss, and a function to evaluate
    the constraints.
    """

    def evaluate_loss(params):
        param_x, param_y = params
        return param_x**2 + 2 * param_y**2

    def evaluate_constraints(params) -> List[cooper.ConstraintState]:
        param_x, param_y = params
        cg0_state = cooper.ConstraintState(violation=-param_x - param_y + 1.0)  # x + y \ge 1
        cg1_state = cooper.ConstraintState(violation=param_x**2 + param_y - 1.0)  # x**2 + y \le 1.0
        return [cg0_state, cg1_state]

    use_ineq_constraints = Toy2dCMP_problem_properties["use_ineq_constraints"]

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers, Toy2dCMP_params_init
    )

    if use_ineq_constraints:
        multiplier_kwargs = {"shape": 1, "device": device}
        constraint_kwargs = {
            "constraint_type": cooper.ConstraintType.INEQUALITY,
            "formulation_type": cooper.FormulationType.LAGRANGIAN,
        }
        cg0 = cooper.ConstraintGroup(**constraint_kwargs, multiplier_kwargs=multiplier_kwargs)
        cg1 = cooper.ConstraintGroup(**constraint_kwargs, multiplier_kwargs=multiplier_kwargs)
        constraint_groups = [cg0, cg1]
    else:
        constraint_groups = []

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(primal_optimizers, constraint_groups)

    for step_id in range(1500):
        cooper_optimizer.zero_grad()

        loss = evaluate_loss(params)

        if use_ineq_constraints:
            cg0_state, cg1_state = evaluate_constraints(params)
            observed_constraints = [(cg0, cg0_state), (cg1, cg1_state)]

            # # Alternatively, one could assign the constraint states directly to the
            # # constraint groups and collect only the constraint groups when gathering
            # # the observed constraints.
            # cg0.state, cg1.state = evaluate_constraints(params)
            # observed_constraints = [cg0, cg1]
        else:
            observed_constraints = []

        cmp_state = cooper.CMPState(loss=loss, observed_constraints=observed_constraints)
        lagrangian_store = cmp_state.populate_lagrangian(return_multipliers=True)
        cmp_state.backward()
        cooper_optimizer.step()

    for param, exact_solution in zip(params, Toy2dCMP_problem_properties["exact_solution"]):
        assert torch.allclose(param, exact_solution)
