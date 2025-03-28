import pytest
import torch

import cooper
import testing


@pytest.fixture
def steps(formulation_type, constraint_type, num_variables):
    if formulation_type == cooper.formulations.QuadraticPenalty:
        if constraint_type == cooper.ConstraintType.INEQUALITY:
            return 45_000
        if constraint_type == cooper.ConstraintType.EQUALITY and num_variables == 10:  # noqa: PLR2004
            return 45_000
    return 1_500


def test_convergence_no_constraint(unconstrained_cmp, params, cooper_optimizer_no_constraint, steps):
    for _ in range(steps):
        cooper_optimizer_no_constraint.roll(compute_cmp_state_kwargs={"x": torch.cat(params)})

    # Compute the exact solution
    x_star, _ = unconstrained_cmp.compute_exact_solution()

    # Check if the primal variable is close to the exact solution
    assert torch.allclose(torch.cat(params), x_star, atol=1e-5)


def test_convergence_with_constraint(
    cmp,
    constraint_params,
    params,
    formulation_type,
    alternation_type,
    cooper_optimizer,
    penalty_updater,
    use_surrogate,
    steps,
):
    for _ in range(steps):
        roll_kwargs = {"compute_cmp_state_kwargs": {"x": torch.cat(params)}}
        if alternation_type == testing.AlternationType.PRIMAL_DUAL:
            roll_kwargs["compute_violations_kwargs"] = {"x": torch.cat(params)}

        roll_out = cooper_optimizer.roll(**roll_kwargs)
        if penalty_updater is not None:
            penalty_updater.step(roll_out.cmp_state.observed_constraints)

    # Compute the exact solution
    x_star, lambda_star = cmp.compute_exact_solution()
    lhs, rhs = constraint_params

    if not use_surrogate:
        # Check if the primal variable is close to the exact solution
        atol = 1e-3 if formulation_type == cooper.formulations.QuadraticPenalty else 1e-4
        assert torch.allclose(torch.cat(params), x_star, atol=atol)

        if formulation_type.expects_multiplier:
            # Check if the dual variable is close to the exact solution.
            # The cmp used only has one constraint.
            assert torch.allclose(next(iter(cmp.dual_parameters())).view(-1), lambda_star[0], atol=atol)
    else:
        # The surrogate formulation is not guaranteed to converge to the exact solution,
        # but it should be feasible
        atol = 5e-4
        assert torch.le(lhs @ torch.cat(params) - rhs, atol).all()
