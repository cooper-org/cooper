import math

import pytest
import torch

import cooper
from cooper.penalty_coefficient_updaters import MultiplicativePenaltyCoefficientUpdater
from tests.helpers import cooper_test_utils

PRIMAL_LR = 3e-2
DUAL_LR = 2e-1
PENALTY_GROWTH_FACTOR = 1.0 + 2.5e-4
PENALTY_VIOLATION_TOLERANCE = 1e-4


@pytest.fixture
def cooper_optimizer_no_constraint(cmp_no_constraint, params):
    primal_optimizers = cooper_test_utils.build_primal_optimizers(
        params, primal_optimizer_kwargs=[{"lr": PRIMAL_LR} for _ in range(len(params))]
    )
    cooper_optimizer = cooper_test_utils.build_cooper_optimizer(
        cmp=cmp_no_constraint, primal_optimizers=primal_optimizers
    )
    return cooper_optimizer


@pytest.fixture
def cooper_optimizer(
    cmp, params, num_variables, use_multiple_primal_optimizers, extrapolation, alternation_type, formulation_type
):
    primal_optimizer_kwargs = [{"lr": PRIMAL_LR}]
    if use_multiple_primal_optimizers:
        primal_optimizer_kwargs.append({"lr": 10 * PRIMAL_LR, "betas": (0.0, 0.0), "eps": 10.0})
    primal_optimizers = cooper_test_utils.build_primal_optimizers(
        params, extrapolation, primal_optimizer_kwargs=primal_optimizer_kwargs
    )

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer(
        cmp=cmp,
        primal_optimizers=primal_optimizers,
        extrapolation=extrapolation,
        augmented_lagrangian=formulation_type == cooper.AugmentedLagrangianFormulation,
        alternation_type=alternation_type,
        dual_optimizer_class=cooper.optim.ExtraSGD if extrapolation else torch.optim.SGD,
        dual_optimizer_kwargs={"lr": DUAL_LR / math.sqrt(num_variables)},
    )
    return cooper_optimizer


@pytest.fixture
def penalty_updater(cmp, formulation_type):
    if formulation_type != cooper.AugmentedLagrangianFormulation:
        return None
    penalty_updater = MultiplicativePenaltyCoefficientUpdater(
        growth_factor=PENALTY_GROWTH_FACTOR, violation_tolerance=PENALTY_VIOLATION_TOLERANCE
    )
    return penalty_updater


def test_convergence_no_constraint(cmp_no_constraint, params, cooper_optimizer_no_constraint):
    for _ in range(2000):
        cooper_optimizer_no_constraint.roll(compute_cmp_state_kwargs=dict(x=torch.cat(params)))

    # Compute the exact solution
    x_star, _ = cmp_no_constraint.compute_exact_solution()

    # Check if the primal variable is close to the exact solution
    assert torch.allclose(torch.cat(params), x_star, atol=1e-5)


def test_convergence_with_constraint(
    cmp, constraint_params, params, alternation_type, formulation_type, cooper_optimizer, penalty_updater, use_surrogate
):
    is_augmented_lagrangian = formulation_type == cooper.AugmentedLagrangianFormulation
    lhs, rhs = constraint_params

    for _ in range(2000):
        roll_kwargs = {"compute_cmp_state_kwargs": dict(x=torch.cat(params))}
        if alternation_type == cooper_test_utils.AlternationType.PRIMAL_DUAL:
            roll_kwargs["compute_violations_kwargs"] = dict(x=torch.cat(params))

        roll_out = cooper_optimizer.roll(**roll_kwargs)
        if is_augmented_lagrangian:
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
