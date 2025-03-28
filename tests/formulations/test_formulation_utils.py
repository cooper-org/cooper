import pytest
import torch

import cooper
import cooper.formulations.utils


@pytest.fixture(params=[1, 100])
def num_constraints(request):
    return request.param


def test_evaluate_constraint_factor_dense_multiplier(num_constraints):
    multiplier = cooper.multipliers.DenseMultiplier(num_constraints=num_constraints)

    result = cooper.formulations.utils.evaluate_constraint_factor(multiplier, None, (num_constraints,))

    assert torch.equal(result, multiplier())
    assert result.requires_grad


def test_evaluate_constraint_factor_indexed_multiplier(num_constraints):
    multiplier = cooper.multipliers.IndexedMultiplier(num_constraints=num_constraints)

    constraint_features = torch.randperm(num_constraints, generator=torch.Generator().manual_seed(0))
    result = cooper.formulations.utils.evaluate_constraint_factor(multiplier, constraint_features, (num_constraints,))

    assert torch.equal(result, multiplier(constraint_features))
    assert result.requires_grad


def test_evaluate_constraint_factor_dense_penalty(num_constraints):
    penalty = cooper.penalty_coefficients.DensePenaltyCoefficient(init=torch.tensor(1.0))

    result = cooper.formulations.utils.evaluate_constraint_factor(penalty, None, (num_constraints,))
    assert torch.equal(result, penalty().expand(num_constraints))
    assert not result.requires_grad


def test_evaluate_constraint_factor_indexed_penalty(num_constraints):
    penalty = cooper.penalty_coefficients.IndexedPenaltyCoefficient(init=torch.tensor(1.0))
    constraint_features = torch.randperm(num_constraints, generator=torch.Generator().manual_seed(0))

    result = cooper.formulations.utils.evaluate_constraint_factor(penalty, constraint_features, (num_constraints,))
    assert torch.equal(result, penalty(constraint_features).expand(num_constraints))
    assert not result.requires_grad


def test_compute_primal_weighted_violation(num_constraints):
    generator = torch.Generator().manual_seed(0)
    constraint_factor_value = torch.rand(num_constraints, generator=generator)
    violation = torch.randn(num_constraints, generator=generator)

    result = cooper.formulations.utils.compute_primal_weighted_violation(constraint_factor_value, violation)
    assert torch.allclose(result, torch.dot(constraint_factor_value, violation))


def test_compute_dual_weighted_violation(num_constraints):
    generator = torch.Generator().manual_seed(0)
    multiplier_value = torch.rand(num_constraints, generator=generator)
    violation = torch.randn(num_constraints, generator=generator)

    result = cooper.formulations.utils.compute_dual_weighted_violation(multiplier_value, violation)
    assert torch.allclose(result, torch.sum(multiplier_value * violation))


@pytest.mark.parametrize("constraint_type", [cooper.ConstraintType.EQUALITY, cooper.ConstraintType.INEQUALITY])
def test_compute_quadratic_penalty(num_constraints, constraint_type):
    generator = torch.Generator().manual_seed(0)
    violation = torch.randn(num_constraints, generator=generator)
    clamped_violation = violation.relu() if constraint_type == cooper.ConstraintType.INEQUALITY else violation
    penalty_coefficient_value = torch.rand(num_constraints, generator=generator)

    result = cooper.formulations.utils.compute_quadratic_penalty(penalty_coefficient_value, violation, constraint_type)
    assert torch.allclose(result, 0.5 * torch.dot(penalty_coefficient_value, clamped_violation**2))


def test_compute_primal_quadratic_augmented_contribution_inequality(num_constraints):
    generator = torch.Generator().manual_seed(0)
    violation = torch.randn(num_constraints, generator=generator)
    multiplier_value = torch.rand(num_constraints, generator=generator)
    penalty_coefficient_value = torch.rand(num_constraints, generator=generator)

    aug_lag = 0.5 * torch.dot(
        1 / penalty_coefficient_value,
        torch.relu(multiplier_value + penalty_coefficient_value * violation) ** 2 - multiplier_value**2,
    )

    result = cooper.formulations.utils.compute_primal_quadratic_augmented_contribution(
        multiplier_value, penalty_coefficient_value, violation, cooper.ConstraintType.INEQUALITY
    )
    assert torch.allclose(result, aug_lag)


def test_compute_primal_quadratic_augmented_contribution_equality(num_constraints):
    generator = torch.Generator().manual_seed(0)
    violation = torch.randn(num_constraints, generator=generator)
    multiplier_value = torch.rand(num_constraints, generator=generator)
    penalty_coefficient_value = torch.rand(num_constraints, generator=generator)

    aug_lag = torch.dot(multiplier_value, violation) + 0.5 * torch.dot(penalty_coefficient_value, violation**2)

    result = cooper.formulations.utils.compute_primal_quadratic_augmented_contribution(
        multiplier_value, penalty_coefficient_value, violation, cooper.ConstraintType.EQUALITY
    )
    assert torch.allclose(result, aug_lag)
