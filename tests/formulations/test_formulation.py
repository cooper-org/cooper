from typing import Literal

import pytest
import torch

import cooper


@pytest.fixture(params=[1, 10])
def num_constraints(request):
    return request.param


@pytest.fixture(params=[cooper.ConstraintType.EQUALITY, cooper.ConstraintType.INEQUALITY])
def constraint_type(request):
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
def constraint_state(num_constraints):
    return cooper.ConstraintState(violation=torch.randn(num_constraints, generator=torch.Generator().manual_seed(0)))


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


def test_formulation_init(constraint_type, formulation_type):
    formulation = formulation_type(constraint_type=constraint_type)
    assert formulation.constraint_type == constraint_type


def test_formulation_init_fails_with_invalid_constraint_type(formulation_type):
    with pytest.raises(ValueError, match=r".*requires either an equality or inequality constraint.*"):
        formulation_type(constraint_type=object())


@pytest.mark.parametrize("primal_or_dual", ["primal", "dual"])
def test_prepare_kwargs_for_lagrangian_contribution(
    primal_or_dual: Literal["primal", "dual"],
    formulation_type,
    constraint_type,
    constraint_state,
    multiplier,
    penalty_coefficient,
):
    # Create an instance of a Formulation
    formulation = formulation_type(constraint_type=constraint_type)

    # Call _prepare_kwargs_for_lagrangian_contribution
    violation, multiplier_value, penalty_coefficient_value = formulation._prepare_kwargs_for_lagrangian_contribution(
        constraint_state=constraint_state,
        multiplier=multiplier,
        penalty_coefficient=penalty_coefficient,
        primal_or_dual=primal_or_dual,
    )

    # Check that the returned values are of the correct type
    assert isinstance(violation, torch.Tensor)

    if formulation.expects_multiplier:
        assert isinstance(multiplier_value, torch.Tensor)
    else:
        assert multiplier_value is None

    if formulation.expects_penalty_coefficient:
        assert isinstance(penalty_coefficient_value, torch.Tensor)
    else:
        assert penalty_coefficient_value is None


def test_formulation_repr(formulation_type, constraint_type):
    formulation = formulation_type(constraint_type=constraint_type)
    assert repr(formulation) == f"{formulation_type.__name__}(constraint_type={constraint_type})"


def test_compute_contribution_to_lagrangian(
    formulation_type, constraint_type, constraint_state, multiplier, penalty_coefficient
):
    # Create an instance of a Formulation
    formulation = formulation_type(constraint_type=constraint_type)

    kwargs = {"constraint_state": constraint_state}
    if formulation.expects_penalty_coefficient:
        kwargs["penalty_coefficient"] = penalty_coefficient
    if formulation.expects_multiplier:
        kwargs["multiplier"] = multiplier

    # Call compute_contribution_to_dual_lagrangian
    dual_contribution_store = formulation.compute_contribution_to_dual_lagrangian(**kwargs)

    # Check that the returned value is of the correct type
    if formulation_type == cooper.formulations.QuadraticPenalty:
        assert dual_contribution_store is None
    else:
        assert isinstance(dual_contribution_store, cooper.formulations.ContributionStore)

    # Call compute_contribution_to_primal_lagrangian
    primal_contribution_store = formulation.compute_contribution_to_primal_lagrangian(**kwargs)
    # Check that the returned value is of the correct type
    assert isinstance(primal_contribution_store, cooper.formulations.ContributionStore)


def test_compute_contribution_to_lagrangian_returns_none_when_constraint_state_does_not_contribute_to_update(
    formulation_type, constraint_type, constraint_state, multiplier, penalty_coefficient
):
    # Create an instance of a Formulation
    formulation = formulation_type(constraint_type=constraint_type)

    kwargs = {"constraint_state": constraint_state}
    if formulation.expects_penalty_coefficient:
        kwargs["penalty_coefficient"] = penalty_coefficient
    if formulation.expects_multiplier:
        kwargs["multiplier"] = multiplier

    # Set contributes_to_dual_update to False
    constraint_state.contributes_to_dual_update = False

    # Call compute_contribution_to_dual_lagrangian
    dual_contribution_store = formulation.compute_contribution_to_dual_lagrangian(**kwargs)

    # Check that the returned value is None
    assert dual_contribution_store is None

    # Set contributes_to_primal_update to False
    constraint_state.contributes_to_primal_update = False

    # Call compute_contribution_to_primal_lagrangian
    primal_contribution_store = formulation.compute_contribution_to_primal_lagrangian(**kwargs)

    # Check that the returned value is None
    assert primal_contribution_store is None
