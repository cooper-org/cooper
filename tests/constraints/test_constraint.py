import pytest
import torch

import cooper


@pytest.fixture(params=[cooper.ConstraintType.EQUALITY, cooper.ConstraintType.INEQUALITY])
def constraint_type(request):
    return request.param


@pytest.fixture(params=[10, 100])
def num_constraints(request):
    return request.param


@pytest.fixture(
    params=[
        cooper.formulations.Lagrangian,
        cooper.formulations.QuadraticPenalty,
        cooper.formulations.AugmentedLagrangian,
    ]
)
def formulation_type(request):
    return request.param


@pytest.fixture
def multiplier(num_constraints, formulation_type):
    if formulation_type.expects_multiplier:
        return cooper.multipliers.DenseMultiplier(num_constraints=num_constraints)
    return None


@pytest.fixture
def penalty_coefficient(num_constraints, formulation_type):
    if formulation_type.expects_penalty_coefficient:
        return cooper.penalty_coefficients.DensePenaltyCoefficient(init=torch.ones(num_constraints))
    return None


@pytest.fixture
def valid_constraint(constraint_type, formulation_type, multiplier, penalty_coefficient):
    return cooper.Constraint(
        constraint_type=constraint_type,
        multiplier=multiplier,
        formulation_type=formulation_type,
        penalty_coefficient=penalty_coefficient,
    )


def test_constraint_initialization(
    valid_constraint, constraint_type, formulation_type, multiplier, penalty_coefficient
):
    # Test successful initialization
    assert valid_constraint.constraint_type == constraint_type
    assert valid_constraint.formulation_type == formulation_type
    assert valid_constraint.multiplier == multiplier
    assert valid_constraint.penalty_coefficient == penalty_coefficient


@pytest.mark.parametrize(
    "formulation_type", [cooper.formulations.AugmentedLagrangian, cooper.formulations.QuadraticPenalty]
)
def test_constraint_sanity_check_penalty_coefficient(constraint_type, formulation_type, multiplier):
    # Test when penalty coefficient is expected but not provided
    with pytest.raises(ValueError, match="expects a penalty coefficient"):
        _ = cooper.Constraint(
            constraint_type=constraint_type,
            multiplier=multiplier,
            formulation_type=formulation_type,
            penalty_coefficient=None,
        )


@pytest.mark.parametrize("formulation_type", [cooper.formulations.Lagrangian])
def test_constraint_sanity_check_penalty_coefficient_unexpected_penalty_coefficient(
    num_constraints, formulation_type, constraint_type, multiplier
):
    # Test when penalty coefficient is not expected but provided
    with pytest.raises(ValueError, match="Received unexpected penalty coefficient"):
        _ = cooper.Constraint(
            constraint_type=constraint_type,
            multiplier=multiplier,
            formulation_type=formulation_type,
            penalty_coefficient=cooper.penalty_coefficients.DensePenaltyCoefficient(init=torch.ones(num_constraints)),
        )


@pytest.mark.parametrize(
    "formulation_type",
    [
        cooper.formulations.QuadraticPenalty,
        cooper.formulations.AugmentedLagrangian,
    ],
)
def test_constraint_sanity_check_penalty_coefficient_negative_penalty_coefficient(
    constraint_type, formulation_type, multiplier
):
    # Test negative penalty coefficient
    with pytest.raises(ValueError, match="must be non-negative"):
        _ = cooper.Constraint(
            constraint_type=constraint_type,
            multiplier=multiplier,
            formulation_type=formulation_type,
            penalty_coefficient=cooper.penalty_coefficients.DensePenaltyCoefficient(init=-torch.ones(10)),
        )


def test_constraint_compute_contribution_to_lagrangian(num_constraints, valid_constraint):
    constraint_state = cooper.ConstraintState(violation=torch.ones(num_constraints))

    primal_contribution = valid_constraint.compute_contribution_to_lagrangian(constraint_state, "primal")
    assert isinstance(primal_contribution, cooper.formulations.ContributionStore)

    dual_contribution = valid_constraint.compute_contribution_to_lagrangian(constraint_state, "dual")
    if valid_constraint.formulation_type == cooper.formulations.QuadraticPenalty:
        assert dual_contribution is None
    else:
        assert isinstance(dual_contribution, cooper.formulations.ContributionStore)


def test_constraint_repr(valid_constraint, formulation_type):
    repr_str = repr(valid_constraint)
    assert "constraint_type=" in repr_str
    assert "formulation=" in repr_str
    if formulation_type.expects_multiplier:
        assert "multiplier=" in repr_str
    if formulation_type.expects_penalty_coefficient:
        assert "penalty_coefficient=" in repr_str
