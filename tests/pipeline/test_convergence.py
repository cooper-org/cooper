import torch

from tests.helpers import cooper_test_utils


def test_convergence_no_constraint(cmp_no_constraint, params, cooper_optimizer_no_constraint):
    for _ in range(2000):
        cooper_optimizer_no_constraint.roll(compute_cmp_state_kwargs=dict(x=torch.cat(params)))

    # Compute the exact solution
    x_star, _ = cmp_no_constraint.compute_exact_solution()

    # Check if the primal variable is close to the exact solution
    assert torch.allclose(torch.cat(params), x_star, atol=1e-5)


def test_convergence_with_constraint(
    cmp, constraint_params, params, alternation_type, cooper_optimizer, penalty_updater, use_surrogate
):
    lhs, rhs = constraint_params

    for _ in range(2000):
        roll_kwargs = {"compute_cmp_state_kwargs": dict(x=torch.cat(params))}
        if alternation_type == cooper_test_utils.AlternationType.PRIMAL_DUAL:
            roll_kwargs["compute_violations_kwargs"] = dict(x=torch.cat(params))

        roll_out = cooper_optimizer.roll(**roll_kwargs)
        if penalty_updater is not None:
            penalty_updater.step(roll_out.cmp_state.observed_constraints)

    # Compute the exact solution
    x_star, lambda_star = cmp.compute_exact_solution()

    if not use_surrogate:
        # Check if the primal variable is close to the exact solution
        atol = 1e-4
        assert torch.allclose(torch.cat(params), x_star, atol=atol)

        # Check if the dual variable is close to the exact solution
        assert torch.allclose(list(cmp.dual_parameters())[0].view(-1), lambda_star[0], atol=atol)
    else:
        # The surrogate formulation is not guaranteed to converge to the exact solution,
        # but it should be feasible
        atol = 5e-4
        assert torch.le(lhs @ torch.cat(params) - rhs, atol).all()
