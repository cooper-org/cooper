import pytest
import torch

import testing
from cooper import penalty_coefficients


@pytest.fixture(params=[1, 100])
def num_constraints(request):
    return request.param


@pytest.fixture
def init_tensor(num_constraints):
    generator = testing.frozen_rand_generator()
    return torch.rand(num_constraints, generator=generator)


@pytest.fixture(params=[penalty_coefficients.DensePenaltyCoefficient, penalty_coefficients.IndexedPenaltyCoefficient])
def penalty_coefficient_class(request):
    return request.param


@pytest.fixture
def penalty_coefficient(penalty_coefficient_class, init_tensor):
    return penalty_coefficient_class(init_tensor)


def evaluate_penalty_coefficient(penalty_coefficient, indices):
    return penalty_coefficient(indices) if penalty_coefficient.expects_constraint_features else penalty_coefficient()


def test_penalty_coefficient_init_and_forward(penalty_coefficient, num_constraints, init_tensor):
    indices = torch.arange(num_constraints, dtype=torch.long)

    assert torch.equal(evaluate_penalty_coefficient(penalty_coefficient, indices), init_tensor)


def test_penalty_coefficient_initialization_failure_with_wrong_shape(penalty_coefficient_class, init_tensor):
    with pytest.raises(ValueError, match="init must either be a scalar or a 1D tensor"):
        penalty_coefficient_class(init_tensor.unsqueeze(-1))


def test_penalty_coefficient_failure_with_grad(penalty_coefficient_class, init_tensor):
    with pytest.raises(ValueError, match=r"PenaltyCoefficient should not require gradients."):
        penalty_coefficient_class(init_tensor.requires_grad_())


def test_penalty_coefficient_failure_with_wrong_shape(penalty_coefficient, init_tensor):
    with pytest.raises(ValueError, match="PenaltyCoefficient does not match existing shape"):
        penalty_coefficient.value = init_tensor.unsqueeze(-1)


def test_penalty_coefficient_sanity_check(penalty_coefficient_class, init_tensor):
    with pytest.raises(ValueError, match=r"All entries of the penalty coefficient must be non-negative."):
        penalty_coefficient_class(-1 * init_tensor)


def test_penalty_coefficient_to(penalty_coefficient):
    penalty_coefficient.to(dtype=torch.float16)
    assert penalty_coefficient.value.dtype == torch.float16


def test_penalty_coefficient_repr(penalty_coefficient):
    assert f"{type(penalty_coefficient).__name__}" in repr(penalty_coefficient)


def test_indexed_penalty_coefficient_forward_invalid_indices(num_constraints, init_tensor):
    penalty_coefficient = penalty_coefficients.IndexedPenaltyCoefficient(init_tensor)
    indices = torch.arange(num_constraints, dtype=torch.float32)

    with pytest.raises(ValueError, match=r"Indices must be of type torch.long."):
        penalty_coefficient(indices)


def test_save_and_load_state_dict(penalty_coefficient_class, penalty_coefficient, num_constraints):
    indices = torch.arange(num_constraints, dtype=torch.long)
    penalty_coefficient_value = evaluate_penalty_coefficient(penalty_coefficient, indices)
    state_dict = penalty_coefficient.state_dict()

    new_penalty_coefficient = penalty_coefficient_class(torch.ones(num_constraints))
    new_penalty_coefficient.load_state_dict(state_dict)
    new_penalty_coefficient_value = evaluate_penalty_coefficient(new_penalty_coefficient, indices)

    assert torch.equal(new_penalty_coefficient_value, penalty_coefficient_value)
