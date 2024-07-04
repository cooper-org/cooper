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

    constraint_features = torch.randperm(num_constraints)
    result = cooper.formulations.utils.evaluate_constraint_factor(multiplier, constraint_features, (num_constraints,))

    assert torch.equal(result, multiplier(constraint_features))
    assert result.requires_grad


def test_evaluate_constraint_factor_dense_penalty(num_constraints):
    penalty = cooper.multipliers.DensePenaltyCoefficient(init=torch.tensor(1.0))

    result = cooper.formulations.utils.evaluate_constraint_factor(penalty, None, (num_constraints,))

    assert torch.equal(result, penalty().expand(num_constraints))
    assert not result.requires_grad


def test_evaluate_constraint_factor_indexed_penalty(num_constraints):
    penalty = cooper.multipliers.IndexedPenaltyCoefficient(init=torch.tensor(1.0))

    constraint_features = torch.randperm(num_constraints)
    result = cooper.formulations.utils.evaluate_constraint_factor(penalty, constraint_features, (num_constraints,))

    assert torch.equal(result, penalty(constraint_features).expand(num_constraints))
    assert not result.requires_grad


def test_compute_primal_weighted_violation(num_constraints):
    # Test compute_primal_weighted_violation

    constraint_factor_value = torch.rand(num_constraints)
    violation = torch.randn(num_constraints)

    result = cooper.formulations.utils.compute_primal_weighted_violation(constraint_factor_value, violation)
    assert torch.allclose(result, torch.dot(constraint_factor_value, violation))
