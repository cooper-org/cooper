import pytest
import torch

import cooper
from cooper.multipliers import MultiplicativePenaltyCoefficientUpdater


@pytest.fixture(params=[1, 100])
def num_constraints(request):
    return request.param


@pytest.fixture(params=[True, False])
def is_scalar(request):
    return request.param


@pytest.fixture
def penalty_coefficient(num_constraints, multiplier_class, is_scalar):
    init = torch.tensor(1.0) if is_scalar else torch.ones(num_constraints)

    if multiplier_class == cooper.multipliers.IndexedMultiplier:
        return cooper.multipliers.IndexedPenaltyCoefficient(init=init)
    return cooper.multipliers.DensePenaltyCoefficient(init=init)


@pytest.fixture
def constraint(num_constraints, constraint_type, multiplier_class, penalty_coefficient):
    constraint = cooper.Constraint(
        constraint_type=constraint_type,
        multiplier=multiplier_class(num_constraints=num_constraints),
        formulation_type=cooper.formulations.AugmentedLagrangian,
        penalty_coefficient=penalty_coefficient,
    )
    return constraint


def test_initialization_with_negative_violation_tolerance():
    with pytest.raises(ValueError, match=r"Violation tolerance must be non-negative."):
        MultiplicativePenaltyCoefficientUpdater(growth_factor=1.01, violation_tolerance=-1e-4)


def test_update_penalty_coefficient(constraint, num_constraints, is_scalar):
    violation = 2e-4 * torch.ones(num_constraints)
    constraint_state = cooper.ConstraintState(
        violation=violation, constraint_features=torch.arange(num_constraints, dtype=torch.long)
    )

    penalty_updater = MultiplicativePenaltyCoefficientUpdater(growth_factor=1.1, violation_tolerance=1e-4)
    penalty_updater.update_penalty_coefficient_(constraint, constraint_state)

    true_penalty = torch.tensor(1.1) if is_scalar else 1.1 * torch.ones(num_constraints)
    assert torch.equal(constraint.penalty_coefficient.value, true_penalty)


def test_update_penalty_coefficient_with_unsupported_penalty_coefficient_type(constraint):
    constraint.penalty_coefficient = object()  # Unsupported penalty coefficient type
    constraint_state = cooper.ConstraintState(violation=torch.tensor(1e-4))

    penalty_updater = MultiplicativePenaltyCoefficientUpdater(growth_factor=1.1, violation_tolerance=1e-4)
    with pytest.raises(TypeError, match=r".*Unsupported penalty coefficient type.*"):
        penalty_updater.update_penalty_coefficient_(constraint, constraint_state)


def test_update_penalty_coefficient_step(constraint, num_constraints, is_scalar):
    violation = 2e-4 * torch.ones(num_constraints)
    constraint_state = cooper.ConstraintState(
        violation=violation, constraint_features=torch.arange(num_constraints, dtype=torch.long)
    )

    penalty_updater = MultiplicativePenaltyCoefficientUpdater(growth_factor=1.1, violation_tolerance=1e-4)
    penalty_updater.step({constraint: constraint_state})

    true_penalty = torch.tensor(1.1) if is_scalar else 1.1 * torch.ones(num_constraints)
    assert torch.equal(constraint.penalty_coefficient.value, true_penalty)
