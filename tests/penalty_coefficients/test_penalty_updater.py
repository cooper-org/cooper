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
        penalty_updater_type(violation_tolerance=VIOLATION_TOLERANCE)


@pytest.mark.parametrize(
    "penalty_updater_type", [AdditivePenaltyCoefficientUpdater, MultiplicativePenaltyCoefficientUpdater]
)
def test_update_penalty_coefficient_with_unsupported_penalty_coefficient_type(constraint, penalty_updater_type):
    constraint.penalty_coefficient = object()  # Unsupported penalty coefficient type
    constraint_state = cooper.ConstraintState(violation=torch.tensor(1e-4))

    penalty_updater = penalty_updater_type(violation_tolerance=VIOLATION_TOLERANCE)
    with pytest.raises(TypeError, match=r".*Unsupported penalty coefficient type.*"):
        penalty_updater.update_penalty_coefficient_(constraint, constraint_state)


@pytest.mark.parametrize(
    "penalty_updater_type", [AdditivePenaltyCoefficientUpdater, MultiplicativePenaltyCoefficientUpdater]
)
@pytest.mark.parametrize("has_restart", [True, False])
def test_update_penalty_coefficient_step(
    penalty_updater_type, constraint, constraint_type, num_constraints, has_restart
):
    penalty_updater_kwargs = {"violation_tolerance": VIOLATION_TOLERANCE, "has_restart": has_restart}
    if penalty_updater_type == AdditivePenaltyCoefficientUpdater:
        penalty_updater_kwargs["increment"] = PENALTY_INCREMENT
    else:
        penalty_updater_kwargs["growth_factor"] = PENALTY_GROWTH_FACTOR

    penalty_updater = penalty_updater_type(**penalty_updater_kwargs)

    # Initial setup: all constraints violate the tolerance
    initial_violations = 2 * VIOLATION_TOLERANCE * torch.ones(num_constraints)
    initial_constraint_state = create_constraint_state(initial_violations, num_constraints)

    # Perform first penalty update
    penalty_updater.update_penalty_coefficient_(constraint, initial_constraint_state)

    # Calculate expected penalty after first update
    if penalty_updater_type == AdditivePenaltyCoefficientUpdater:
        expected_penalty = constraint.penalty_coefficient.init + PENALTY_INCREMENT
    else:
        expected_penalty = constraint.penalty_coefficient.init * PENALTY_GROWTH_FACTOR

    # Verify first update
    assert torch.equal(constraint.penalty_coefficient.value, expected_penalty)

    # Second update: constraints are now satisfied
    updated_violations = 0.0 * initial_violations
    updated_constraint_state = create_constraint_state(updated_violations, num_constraints)

    # Perform second penalty update
    penalty_updater.update_penalty_coefficient_(constraint, updated_constraint_state)

    # Calculate expected penalty after second update
    if has_restart and constraint_type == cooper.ConstraintType.INEQUALITY:
        # Reset the penalty to the initial value for the first constraint which is now satisfied,
        expected_penalty = constraint.penalty_coefficient.init

    # Final assertion
    assert torch.equal(constraint.penalty_coefficient.value, expected_penalty)
