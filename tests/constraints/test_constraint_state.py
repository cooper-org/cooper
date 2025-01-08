from dataclasses import is_dataclass

import pytest
import torch

import cooper
import testing


@pytest.fixture(params=[1, 10])
def num_constraints(request):
    return request.param


@pytest.fixture
def violation(num_constraints):
    violation = torch.randn(num_constraints, generator=testing.frozen_rand_generator(0))
    if num_constraints == 1:
        violation.squeeze_()
    return violation


@pytest.fixture
def strict_violation(num_constraints):
    strict_violation = torch.randn(num_constraints, generator=testing.frozen_rand_generator(1))
    if num_constraints == 1:
        strict_violation.squeeze_()
    return strict_violation


@pytest.fixture
def constraint_features(num_constraints):
    return torch.randperm(num_constraints, generator=testing.frozen_rand_generator(2))


@pytest.fixture
def strict_constraint_features(num_constraints):
    return torch.randperm(num_constraints, generator=testing.frozen_rand_generator(3))


@pytest.fixture(params=[True, False])
def contributes_to_primal_update(request):
    return request.param


@pytest.fixture(params=[True, False])
def contributes_to_dual_update(request):
    return request.param


@pytest.fixture
def constraint_state(
    violation,
    strict_violation,
    constraint_features,
    strict_constraint_features,
    contributes_to_primal_update,
    contributes_to_dual_update,
):
    return cooper.ConstraintState(
        violation=violation,
        strict_violation=strict_violation,
        constraint_features=constraint_features,
        strict_constraint_features=strict_constraint_features,
        contributes_to_primal_update=contributes_to_primal_update,
        contributes_to_dual_update=contributes_to_dual_update,
    )


def test_constraint_state_initialization(
    constraint_state,
    violation,
    strict_violation,
    constraint_features,
    strict_constraint_features,
    contributes_to_primal_update,
    contributes_to_dual_update,
):
    assert is_dataclass(constraint_state)
    assert torch.equal(constraint_state.violation, violation)
    assert torch.equal(constraint_state.strict_violation, strict_violation)
    assert torch.equal(constraint_state.constraint_features, constraint_features)
    assert torch.equal(constraint_state.strict_constraint_features, strict_constraint_features)
    assert constraint_state.contributes_to_primal_update == contributes_to_primal_update
    assert constraint_state.contributes_to_dual_update == contributes_to_dual_update


def test_constraint_state_initialization_failure(violation, strict_constraint_features):
    with pytest.raises(
        ValueError, match=r"`strict_violation` must be provided if `strict_constraint_features` is provided."
    ):
        cooper.ConstraintState(violation=violation, strict_constraint_features=strict_constraint_features)


def test_extract_violations_without_do_unsqueeze(constraint_state):
    violation, strict_violation = constraint_state.extract_violations(do_unsqueeze=False)
    assert torch.equal(violation, constraint_state.violation)
    assert torch.equal(strict_violation, constraint_state.strict_violation)


def test_extract_violations_with_do_unsqueeze(constraint_state):
    violation, strict_violation = constraint_state.extract_violations()
    assert violation.dim() > 0
    assert torch.equal(violation.view(-1), constraint_state.violation.view(-1))
    assert strict_violation.dim() > 0
    assert torch.equal(strict_violation.view(-1), constraint_state.strict_violation.view(-1))


def test_extract_violations_without_strict_violation(violation):
    # Test when strict_violation is not provided
    constraint_state = cooper.ConstraintState(violation=violation)
    violation, strict_violation = constraint_state.extract_violations(do_unsqueeze=False)
    assert torch.equal(violation, constraint_state.violation)
    assert torch.equal(strict_violation, constraint_state.violation)


def test_extract_constraint_features(constraint_state):
    constraint_features, strict_constraint_features = constraint_state.extract_constraint_features()
    assert torch.equal(constraint_features, constraint_state.constraint_features)
    assert torch.equal(strict_constraint_features, constraint_state.strict_constraint_features)


def test_extract_constraint_features_without_strict_constraint_features(violation, constraint_features):
    # Test when strict_constraint_features is not provided
    constraint_state = cooper.ConstraintState(violation=violation, constraint_features=constraint_features)
    constraint_features, strict_constraint_features = constraint_state.extract_constraint_features()
    assert torch.equal(constraint_features, constraint_state.constraint_features)
    assert torch.equal(strict_constraint_features, constraint_state.constraint_features)


def test_extract_constraint_features_without_constraint_features(violation):
    constraint_state = cooper.ConstraintState(violation=violation)
    constraint_features, strict_constraint_features = constraint_state.extract_constraint_features()
    assert constraint_features is None
    assert strict_constraint_features is None
