from dataclasses import is_dataclass

import pytest
import torch

import cooper


@pytest.fixture(
    params=[
        cooper.formulations.Lagrangian,
        cooper.formulations.AugmentedLagrangian,
        cooper.formulations.QuadraticPenalty,
    ]
)
def formulation_type(request):
    return request.param


@pytest.fixture
def penalty_coefficient(formulation_type):
    if formulation_type.expects_penalty_coefficient:
        return cooper.penalty_coefficients.DensePenaltyCoefficient(torch.ones(1))
    return None


@pytest.fixture
def multiplier(formulation_type):
    if formulation_type.expects_multiplier:
        return cooper.multipliers.DenseMultiplier(num_constraints=1)
    return None


@pytest.fixture
def eq_constraint(formulation_type, multiplier, penalty_coefficient):
    constraint = cooper.Constraint(
        constraint_type=cooper.ConstraintType.EQUALITY,
        formulation_type=formulation_type,
        multiplier=multiplier,
        penalty_coefficient=penalty_coefficient,
    )
    return constraint


@pytest.fixture
def ineq_constraint(formulation_type, multiplier, penalty_coefficient):
    constraint = cooper.Constraint(
        constraint_type=cooper.ConstraintType.INEQUALITY,
        formulation_type=formulation_type,
        multiplier=multiplier,
        penalty_coefficient=penalty_coefficient,
    )
    return constraint


@pytest.fixture
def cmp_state():
    return cooper.CMPState()


@pytest.fixture
def cmp_instance(cmp_state):
    class DummyCMP(cooper.ConstrainedMinimizationProblem):
        def compute_cmp_state(self):  # noqa: PLR6301
            return cmp_state

    return DummyCMP()


def test_lagrangian_store_initialization():
    lagrangian_store = cooper.LagrangianStore()
    assert is_dataclass(lagrangian_store)
    assert lagrangian_store.lagrangian is None
    assert isinstance(lagrangian_store.multiplier_values, dict)
    assert isinstance(lagrangian_store.penalty_coefficient_values, dict)


def test_lagrangian_store_backward():
    lagrangian_store = cooper.LagrangianStore(lagrangian=torch.tensor(1.0, requires_grad=True))
    lagrangian_store.backward()
    assert lagrangian_store.lagrangian.grad is not None


def test_lagrangian_store_backward_none():
    lagrangian_store = cooper.LagrangianStore()
    lagrangian_store.backward()
    assert True  # No error should be raised


def test_lagrangian_store_observed_multiplier_values(eq_constraint, ineq_constraint):
    lagrangian_store = cooper.LagrangianStore(
        multiplier_values={eq_constraint: torch.tensor(1.0), ineq_constraint: torch.tensor(2.0)}
    )
    observed_values = list(lagrangian_store.observed_multiplier_values())
    assert observed_values == [torch.tensor(1.0), torch.tensor(2.0)]


def test_lagrangian_store_observed_penalty_coefficient_values(eq_constraint, ineq_constraint):
    lagrangian_store = cooper.LagrangianStore(
        penalty_coefficient_values={eq_constraint: torch.tensor(3.0), ineq_constraint: torch.tensor(4.0)}
    )
    observed_values = list(lagrangian_store.observed_penalty_coefficient_values())
    assert observed_values == [torch.tensor(3.0), torch.tensor(4.0)]


def test_initial_cmp_state(cmp_state):
    assert is_dataclass(cmp_state)
    assert cmp_state.loss is None
    assert cmp_state.observed_constraints == {}
    assert cmp_state.misc is None


def test_compute_primal_lagrangian_no_constraints_no_loss(cmp_state):
    lagrangian_store = cmp_state.compute_primal_lagrangian()
    assert isinstance(lagrangian_store, cooper.LagrangianStore)
    assert lagrangian_store.lagrangian is None
    assert lagrangian_store.multiplier_values == {}
    assert lagrangian_store.penalty_coefficient_values == {}


def test_compute_dual_lagrangian_no_constraints_no_loss(cmp_state):
    lagrangian_store = cmp_state.compute_dual_lagrangian()
    assert isinstance(lagrangian_store, cooper.LagrangianStore)
    assert lagrangian_store.lagrangian is None
    assert lagrangian_store.multiplier_values == {}
    assert lagrangian_store.penalty_coefficient_values == {}


def test_compute_primal_lagrangian_with_loss(cmp_state):
    cmp_state.loss = torch.tensor(1.0)
    lagrangian_store = cmp_state.compute_primal_lagrangian()
    assert isinstance(lagrangian_store, cooper.LagrangianStore)
    assert lagrangian_store.lagrangian == 1.0


def test_compute_primal_lagrangian_with_constraints(cmp_state, eq_constraint):
    if eq_constraint.multiplier is not None:
        eq_constraint.multiplier.weight.data.fill_(2.0)
    constraint_state = cooper.ConstraintState(violation=torch.tensor(3.0))
    cmp_state.observed_constraints[eq_constraint] = constraint_state
    lagrangian_store = cmp_state.compute_primal_lagrangian()

    true_lagrangian = 0.0
    if eq_constraint.multiplier is not None:
        true_lagrangian += 3.0 * 2.0
    if eq_constraint.penalty_coefficient is not None:
        true_lagrangian += 0.5 * (3.0**2)
    assert lagrangian_store.lagrangian.item() == true_lagrangian


def test_named_observed_violations(cmp_state, eq_constraint):
    constraint_state = cooper.ConstraintState(violation=torch.tensor(3.0))
    cmp_state.observed_constraints[eq_constraint] = constraint_state
    observed_violations = [_[1] for _ in cmp_state.named_observed_violations()]
    assert len(observed_violations) == 1
    assert torch.equal(observed_violations[0], torch.tensor(3.0))


def test_named_observed_strict_violations(cmp_state, eq_constraint):
    constraint_state = cooper.ConstraintState(violation=torch.tensor(0.0), strict_violation=torch.tensor(2.0))
    cmp_state.observed_constraints[eq_constraint] = constraint_state
    strict_violations = [_[1] for _ in cmp_state.named_observed_strict_violations()]
    assert len(strict_violations) == 1
    assert strict_violations[0] == torch.tensor(2.0)


def test_named_observed_constraint_features(cmp_state, eq_constraint):
    constraint_state = cooper.ConstraintState(
        violation=torch.tensor(0.0), constraint_features=torch.tensor(0, dtype=torch.long)
    )
    cmp_state.observed_constraints[eq_constraint] = constraint_state
    constraint_features = [_[1] for _ in cmp_state.named_observed_constraint_features()]
    assert len(constraint_features) == 1
    assert constraint_features[0].item() == 0


def test_named_observed_strict_constraint_features(cmp_state, eq_constraint):
    constraint_state = cooper.ConstraintState(
        violation=torch.tensor(0.0),
        strict_violation=torch.tensor(2.0),
        strict_constraint_features=torch.tensor(0, dtype=torch.long),
    )
    cmp_state.observed_constraints[eq_constraint] = constraint_state
    strict_constraint_features = [_[1] for _ in cmp_state.named_observed_strict_constraint_features()]
    assert len(strict_constraint_features) == 1
    assert strict_constraint_features[0].item() == 0


def test_cmp_register_constraint(cmp_instance, eq_constraint):
    cmp_instance._register_constraint("test_constraint", eq_constraint)
    assert "test_constraint" in cmp_instance._constraints
    assert cmp_instance._constraints["test_constraint"] is eq_constraint


def test_cmp_register_duplicate_constraint(cmp_instance, eq_constraint):
    cmp_instance._register_constraint("test_constraint", eq_constraint)
    with pytest.raises(ValueError, match=r".*already exists.*"):
        cmp_instance._register_constraint("test_constraint", eq_constraint)


def test_cmp_register_constraint_wrong_type(cmp_instance):
    with pytest.raises(TypeError, match="Expected a Constraint instance"):
        cmp_instance._register_constraint("test_constraint", object())


def test_cmp_constraints(cmp_instance, eq_constraint):
    cmp_instance._register_constraint("test_constraint", eq_constraint)
    constraints = list(cmp_instance.constraints())
    assert len(constraints) == 1
    assert constraints[0] is eq_constraint


def test_cmp_named_constraints(cmp_instance, eq_constraint):
    cmp_instance._register_constraint("test_constraint", eq_constraint)
    named_constraints = list(cmp_instance.named_constraints())
    assert len(named_constraints) == 1
    assert named_constraints[0] == ("test_constraint", eq_constraint)


def test_cmp_multipliers(cmp_instance, eq_constraint):
    cmp_instance._register_constraint("test_constraint", eq_constraint)
    multipliers = list(cmp_instance.multipliers())
    if eq_constraint.multiplier is None:
        assert len(multipliers) == 0
    else:
        assert len(multipliers) == 1
        assert multipliers[0] == eq_constraint.multiplier


def test_cmp_named_multipliers(cmp_instance, eq_constraint):
    cmp_instance._register_constraint("test_constraint", eq_constraint)
    named_multipliers = list(cmp_instance.named_multipliers())
    if eq_constraint.multiplier is None:
        assert len(named_multipliers) == 0
    else:
        assert len(named_multipliers) == 1
        assert named_multipliers[0] == ("test_constraint", eq_constraint.multiplier)


def test_cmp_penalty_coefficients(cmp_instance, eq_constraint):
    cmp_instance._register_constraint("test_constraint", eq_constraint)
    penalty_coefficients = list(cmp_instance.penalty_coefficients())
    if eq_constraint.penalty_coefficient is None:
        assert len(penalty_coefficients) == 0
    else:
        assert len(penalty_coefficients) == 1
        assert penalty_coefficients[0] == eq_constraint.penalty_coefficient


def test_cmp_named_penalty_coefficients(cmp_instance, eq_constraint):
    cmp_instance._register_constraint("test_constraint", eq_constraint)
    named_penalty_coefficients = list(cmp_instance.named_penalty_coefficients())
    if eq_constraint.penalty_coefficient is None:
        assert len(named_penalty_coefficients) == 0
    else:
        assert len(named_penalty_coefficients) == 1
        assert named_penalty_coefficients[0] == ("test_constraint", eq_constraint.penalty_coefficient)


def test_cmp_dual_parameters(cmp_instance, eq_constraint):
    cmp_instance._register_constraint("test_constraint", eq_constraint)
    dual_parameters = list(cmp_instance.dual_parameters())
    if eq_constraint.multiplier is None:
        assert len(dual_parameters) == 0
    else:
        expected_parameters = list(eq_constraint.multiplier.parameters())
        assert len(dual_parameters) == len(expected_parameters)


def test_cmp_to(cmp_instance, eq_constraint):
    cmp_instance._register_constraint("test_constraint", eq_constraint)
    cmp_instance.to(torch.float64)

    if eq_constraint.multiplier is not None:
        assert cmp_instance._constraints["test_constraint"].multiplier.weight.dtype == torch.float64
    if eq_constraint.penalty_coefficient is not None:
        assert cmp_instance._constraints["test_constraint"].penalty_coefficient.value.dtype == torch.float64


def test_cmp_state_dict(cmp_instance, eq_constraint):
    cmp_instance._register_constraint("test_constraint", eq_constraint)
    state_dict = cmp_instance.state_dict()
    if eq_constraint.multiplier is not None:
        assert state_dict["multipliers"]["test_constraint"] == eq_constraint.multiplier.state_dict()
    if eq_constraint.penalty_coefficient is not None:
        assert state_dict["penalty_coefficients"]["test_constraint"] == eq_constraint.penalty_coefficient.state_dict()


def test_load_state_dict(cmp_instance, eq_constraint):
    cmp_instance._register_constraint("test_constraint", eq_constraint)
    state_dict = cmp_instance.state_dict()
    cmp_instance.load_state_dict(state_dict)

    if eq_constraint.multiplier is not None:
        assert (
            cmp_instance._constraints["test_constraint"].multiplier.state_dict()
            == state_dict["multipliers"]["test_constraint"]
        )

    if eq_constraint.penalty_coefficient is not None:
        assert (
            cmp_instance._constraints["test_constraint"].penalty_coefficient.state_dict()
            == state_dict["penalty_coefficients"]["test_constraint"]
        )


def test_cmp_setattr_getattr_delattr(cmp_instance, eq_constraint):
    cmp_instance.test_constraint = eq_constraint
    cmp_instance.tolerance = 1e-2
    assert cmp_instance.test_constraint == eq_constraint
    del cmp_instance.test_constraint
    with pytest.raises(AttributeError):
        _ = cmp_instance.test_constraint
    del cmp_instance.tolerance


def test_repr(cmp_instance, eq_constraint):
    cmp_instance._register_constraint("test_constraint", eq_constraint)
    repr_str = repr(cmp_instance)
    assert "test_constraint" in repr_str
    assert cmp_instance.__class__.__name__ in repr_str


def test_sanity_check_cmp_state_loss(cmp_instance):
    with pytest.raises(ValueError, match=r"The loss tensor must have a valid gradient."):
        cmp_instance.sanity_check_cmp_state(cooper.CMPState(loss=torch.tensor(1.0)))


def test_sanity_check_cmp_state_violation(cmp_instance, eq_constraint):
    cmp_instance._register_constraint("test_constraint", eq_constraint)
    with pytest.raises(ValueError, match=r"The violation tensor of constraint .*"):
        cmp_instance.sanity_check_cmp_state(
            cooper.CMPState(observed_constraints={eq_constraint: cooper.ConstraintState(violation=torch.tensor(1.0))})
        )


def test_sanity_check_cmp_state_strict_violation(cmp_instance, eq_constraint):
    cmp_instance._register_constraint("test_constraint", eq_constraint)
    violation = torch.tensor(1.0, requires_grad=True)
    violation.backward()
    with pytest.raises(ValueError, match=r".must not have a gradient."):
        cmp_instance.sanity_check_cmp_state(
            cooper.CMPState(
                observed_constraints={
                    eq_constraint: cooper.ConstraintState(violation=violation, strict_violation=violation)
                }
            )
        )


def test_sanity_check_cmp_state_contributing_to_dual_update(cmp_instance):
    test_constraint = cooper.Constraint(
        constraint_type=cooper.ConstraintType.EQUALITY,
        formulation_type=cooper.formulations.QuadraticPenalty,
        penalty_coefficient=cooper.penalty_coefficients.DensePenaltyCoefficient(torch.ones(1)),
    )
    cmp_instance._register_constraint("test_constraint", test_constraint)
    violation = torch.tensor(1.0, requires_grad=True)
    violation.backward()
    with pytest.raises(ValueError, match=r"ConstraintState contributes to dual update.*"):
        cmp_instance.sanity_check_cmp_state(
            cooper.CMPState(observed_constraints={test_constraint: cooper.ConstraintState(violation=violation)})
        )
