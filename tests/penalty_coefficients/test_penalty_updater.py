import pytest
import torch

import cooper
from cooper.penalty_coefficients import AdditivePenaltyCoefficientUpdater, MultiplicativePenaltyCoefficientUpdater

VIOLATION_TOLERANCE = 1e-4
PENALTY_INCREMENT = 1.1
PENALTY_GROWTH_FACTOR = 1.1


@pytest.fixture(params=[1, 100])
def num_constraints(request):
    return request.param


@pytest.fixture(params=[True, False])
def is_scalar(request):
    return request.param


@pytest.fixture(
    params=[cooper.penalty_coefficients.DensePenaltyCoefficient, cooper.penalty_coefficients.IndexedPenaltyCoefficient]
)
def penalty_coefficient(request, num_constraints, is_scalar):
    penalty_coefficient_type = request.param
    init = torch.tensor(1.0) if is_scalar else torch.ones(num_constraints)
    return penalty_coefficient_type(init=init)


@pytest.fixture(params=[cooper.ConstraintType.EQUALITY, cooper.ConstraintType.INEQUALITY])
def constraint_type(request):
    return request.param


@pytest.fixture
def constraint(constraint_type, penalty_coefficient):
    constraint = cooper.Constraint(
        constraint_type=constraint_type,
        formulation_type=cooper.formulations.QuadraticPenalty,
        penalty_coefficient=penalty_coefficient,
    )
    return constraint


def create_constraint_state(violations, num_constraints):
    return cooper.ConstraintState(
        violation=violations,
        constraint_features=torch.arange(num_constraints, dtype=torch.long),
    )


@pytest.mark.parametrize(
    "penalty_updater_type", [AdditivePenaltyCoefficientUpdater, MultiplicativePenaltyCoefficientUpdater]
)
def test_initialization_with_negative_violation_tolerance(penalty_updater_type):
    with pytest.raises(ValueError, match=r"Violation tolerance must be non-negative."):
        penalty_updater_type(violation_tolerance=-1e-4)


@pytest.mark.parametrize(
    "penalty_updater_type", [AdditivePenaltyCoefficientUpdater, MultiplicativePenaltyCoefficientUpdater]
)
def test_update_penalty_coefficient_with_unsupported_penalty_coefficient_type(constraint, penalty_updater_type):
    constraint.penalty_coefficient = object()  # Unsupported penalty coefficient type
    constraint_state = cooper.ConstraintState(violation=torch.tensor(1e-4))

    penalty_updater = penalty_updater_type(violation_tolerance=1e-4)
    with pytest.raises(TypeError, match=r".*Unsupported penalty coefficient type.*"):
        penalty_updater.update_penalty_coefficient_(constraint, constraint_state)


def test_multiplicative_update_penalty_coefficient_step(constraint, num_constraints):
    penalty_updater = MultiplicativePenaltyCoefficientUpdater(
        growth_factor=PENALTY_GROWTH_FACTOR, violation_tolerance=VIOLATION_TOLERANCE
    )

    violation = 2e-4 * torch.ones(num_constraints)
    constraint_state = create_constraint_state(violation, num_constraints)
    penalty_updater.update_penalty_coefficient_(constraint, constraint_state)

    expected_penalty = PENALTY_GROWTH_FACTOR * constraint.penalty_coefficient.init
    assert torch.equal(constraint.penalty_coefficient.value, expected_penalty)


@pytest.mark.parametrize("has_restart", [True, False])
def test_additive_update_penalty_coefficient_step(constraint, constraint_type, num_constraints, is_scalar, has_restart):
    penalty_updater = AdditivePenaltyCoefficientUpdater(
        increment=PENALTY_INCREMENT, violation_tolerance=VIOLATION_TOLERANCE, has_restart=has_restart
    )

    # Initial setup: all constraints violate the threshold
    initial_violations = 2 * VIOLATION_TOLERANCE * torch.ones(num_constraints)
    initial_constraint_state = create_constraint_state(initial_violations, num_constraints)

    # Perform first penalty update
    penalty_updater.update_penalty_coefficient_(constraint, initial_constraint_state)

    # Verify first update
    expected_penalty = constraint.penalty_coefficient.init + PENALTY_INCREMENT
    assert torch.equal(constraint.penalty_coefficient.value, expected_penalty)

    # Second update: first constraint is now satisfied
    updated_violations = initial_violations.clone()
    updated_violations[0] = 0.0
    updated_constraint_state = create_constraint_state(updated_violations, num_constraints)

    # Perform second penalty update
    penalty_updater.update_penalty_coefficient_(constraint, updated_constraint_state)

    # Calculate expected penalty after second update
    if is_scalar:
        # Handle scalar penalty case
        violation_norm = updated_violations.norm()

        # Apply penalty increment if violation exceeds the threshold
        if violation_norm > VIOLATION_TOLERANCE:
            expected_penalty += PENALTY_INCREMENT

        # Reset the penalty to the initial value if an inequality constraint is satisfied
        if has_restart and constraint_type == cooper.ConstraintType.INEQUALITY and violation_norm == 0:
            expected_penalty = constraint.penalty_coefficient.init
    elif has_restart and constraint_type == cooper.ConstraintType.INEQUALITY:
        # Reset the penalty to the initial value for the first constraint which is now satisfied,
        # and apply the penalty increment for the remaining constraints.
        expected_penalty += PENALTY_INCREMENT
        expected_penalty[0] = constraint.penalty_coefficient.init[0]
    else:
        # Increment the penalty for all constraints that exceed the violation threshold.
        # Skip the increment for the first constraint as it is now satisfied.
        original_first_element = expected_penalty[0]
        expected_penalty = expected_penalty + PENALTY_INCREMENT
        expected_penalty[0] = original_first_element

    # Final assertion
    assert torch.equal(constraint.penalty_coefficient.value, expected_penalty)
