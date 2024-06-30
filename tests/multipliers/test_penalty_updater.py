import pytest
import torch

import cooper
from cooper.penalty_coefficient_updaters import MultiplicativePenaltyCoefficientUpdater


def test_initialization_with_negative_violation_tolerance():
    with pytest.raises(ValueError, match="Violation tolerance must be non-negative."):
        MultiplicativePenaltyCoefficientUpdater(growth_factor=1.01, violation_tolerance=-1e-4)


@pytest.mark.parametrize("constraint", ["eq_constraint", "ineq_constraint"])
def test_update_penalty_coefficient_with_dense_penalty(constraint, request):
    constraint = request.getfixturevalue(constraint)
    constraint.penalty_coefficient = cooper.multipliers.DensePenaltyCoefficient(torch.tensor(1.0))
    constraint_state = cooper.ConstraintState(violation=torch.tensor(2e-4))

    penalty_updater = MultiplicativePenaltyCoefficientUpdater(growth_factor=1.1, violation_tolerance=1e-4)
    penalty_updater.update_penalty_coefficient_(constraint, constraint_state)

    assert torch.equal(constraint.penalty_coefficient.value, torch.tensor([1.1]))


@pytest.mark.parametrize("constraint", ["eq_constraint", "ineq_constraint"])
def test_update_penalty_coefficient_with_indexed_penalty(constraint, request):
    constraint = request.getfixturevalue(constraint)
    constraint.penalty_coefficient = cooper.multipliers.IndexedPenaltyCoefficient(torch.tensor(1.0))
    constraint_state = cooper.ConstraintState(
        violation=torch.tensor(2e-4), constraint_features=torch.tensor(0, dtype=torch.long)
    )

    penalty_updater = MultiplicativePenaltyCoefficientUpdater(growth_factor=1.1, violation_tolerance=1e-4)
    penalty_updater.update_penalty_coefficient_(constraint, constraint_state)

    assert torch.equal(constraint.penalty_coefficient.value, torch.tensor([1.1]))


@pytest.mark.parametrize("constraint", ["eq_constraint", "ineq_constraint"])
def test_update_penalty_coefficient_when_violation_below_tolerance_no_update(constraint, request):
    constraint = request.getfixturevalue(constraint)
    constraint.penalty_coefficient = cooper.multipliers.DensePenaltyCoefficient(torch.tensor(1.0))
    constraint_state = cooper.ConstraintState(violation=torch.tensor(1e-4))

    penalty_updater = MultiplicativePenaltyCoefficientUpdater(growth_factor=1.1, violation_tolerance=1e-2)
    penalty_updater.update_penalty_coefficient_(constraint, constraint_state)

    assert torch.equal(constraint.penalty_coefficient.value, torch.tensor([1.0]))


@pytest.mark.parametrize("constraint", ["eq_constraint", "ineq_constraint"])
def test_update_penalty_coefficient_with_unsupported_penalty_coefficient_type(constraint, request):
    constraint = request.getfixturevalue(constraint)
    constraint.penalty_coefficient = object()  # Unsupported penalty coefficient type
    constraint_state = cooper.ConstraintState(violation=torch.tensor(1e-4))

    penalty_updater = MultiplicativePenaltyCoefficientUpdater(growth_factor=1.1, violation_tolerance=1e-4)
    with pytest.raises(ValueError, match=r".*Unsupported penalty coefficient type.*"):
        penalty_updater.update_penalty_coefficient_(constraint, constraint_state)
