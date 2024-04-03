import cooper_test_utils
import torch


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

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(primal_optimizers, cmp=cmp)

    for step_id in range(1500):
        cmp_state, lagrangian_store = cooper_optimizer.roll(compute_cmp_state_kwargs=dict(params=params))

    for param, exact_solution in zip(params, Toy2dCMP_problem_properties["exact_solution"]):
        assert torch.allclose(param, exact_solution)
