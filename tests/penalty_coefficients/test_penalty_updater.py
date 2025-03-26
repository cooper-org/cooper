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


@pytest.fixture
def penalty_updater():
    class DummyPenaltyCoefficientUpdater(cooper.penalty_coefficients.PenaltyCoefficientUpdater):
        def update_penalty_coefficient_(self, constraint, constraint_state):  # noqa: ARG002, PLR6301
            constraint.penalty_coefficient.value = PENALTY_GROWTH_FACTOR * constraint.penalty_coefficient.value

    return DummyPenaltyCoefficientUpdater()


@pytest.mark.parametrize("contributes_to_primal_update", [True, False])
def test_updater_step_expects_multiplier_contributes_to_dual(penalty_updater, contributes_to_primal_update):
    multiplier = cooper.multipliers.DenseMultiplier(num_constraints=1)
    penalty_coefficient = cooper.penalty_coefficients.DensePenaltyCoefficient(init=torch.tensor(1.0))
    constraint = cooper.Constraint(
        constraint_type=cooper.ConstraintType.EQUALITY,
        formulation_type=cooper.formulations.AugmentedLagrangian,
        multiplier=multiplier,
        penalty_coefficient=penalty_coefficient,
    )
    constraint_state = cooper.ConstraintState(
        violation=torch.tensor(1e-4), contributes_to_primal_update=contributes_to_primal_update
    )

    penalty_updater.step({constraint: constraint_state})
    assert torch.all(torch.eq(penalty_coefficient.value, PENALTY_GROWTH_FACTOR * penalty_coefficient.init))


@pytest.mark.parametrize("contributes_to_primal_update", [True, False])
def test_updater_step_expects_multiplier_not_contributes_to_dual_without_strict_violation(
    penalty_updater, contributes_to_primal_update
):
    multiplier = cooper.multipliers.DenseMultiplier(num_constraints=1)
    penalty_coefficient = cooper.penalty_coefficients.DensePenaltyCoefficient(init=torch.tensor(1.0))
    constraint = cooper.Constraint(
        constraint_type=cooper.ConstraintType.EQUALITY,
        formulation_type=cooper.formulations.AugmentedLagrangian,
        multiplier=multiplier,
        penalty_coefficient=penalty_coefficient,
    )
    constraint_state = cooper.ConstraintState(
        violation=torch.tensor(1e-4),
        contributes_to_dual_update=False,
        contributes_to_primal_update=contributes_to_primal_update,
    )

    penalty_updater.step({constraint: constraint_state})
    assert torch.all(torch.eq(penalty_coefficient.value, penalty_coefficient.init))


def test_updater_step_expects_multiplier_not_contributes_to_dual_with_strict_violation(penalty_updater):
    multiplier = cooper.multipliers.DenseMultiplier(num_constraints=1)
    penalty_coefficient = cooper.penalty_coefficients.DensePenaltyCoefficient(init=torch.tensor(1.0))
    constraint = cooper.Constraint(
        constraint_type=cooper.ConstraintType.EQUALITY,
        formulation_type=cooper.formulations.AugmentedLagrangian,
        multiplier=multiplier,
        penalty_coefficient=penalty_coefficient,
    )
    constraint_state = cooper.ConstraintState(
        violation=torch.tensor(1e-4),
        strict_violation=torch.tensor(1e-4),
        contributes_to_dual_update=False,
        contributes_to_primal_update=True,
    )

    penalty_updater.step({constraint: constraint_state})
    assert torch.all(torch.eq(penalty_coefficient.value, PENALTY_GROWTH_FACTOR * penalty_coefficient.init))


def test_updater_step_not_expects_multiplier_contributes_to_primal(penalty_updater):
    penalty_coefficient = cooper.penalty_coefficients.DensePenaltyCoefficient(init=torch.tensor(1.0))
    constraint = cooper.Constraint(
        constraint_type=cooper.ConstraintType.EQUALITY,
        formulation_type=cooper.formulations.QuadraticPenalty,
        penalty_coefficient=penalty_coefficient,
    )
    constraint_state = cooper.ConstraintState(
        violation=torch.tensor(1e-4), contributes_to_dual_update=False, contributes_to_primal_update=True
    )

    penalty_updater.step({constraint: constraint_state})
    assert torch.all(torch.eq(penalty_coefficient.value, PENALTY_GROWTH_FACTOR * penalty_coefficient.init))


@pytest.mark.parametrize(
    "penalty_updater_type", [AdditivePenaltyCoefficientUpdater, MultiplicativePenaltyCoefficientUpdater]
)
def test_initialization_with_negative_violation_tolerance(penalty_updater_type):
    with pytest.raises(ValueError, match=r"Violation tolerance must be non-negative."):
        penalty_updater_type(violation_tolerance=-VIOLATION_TOLERANCE)


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
