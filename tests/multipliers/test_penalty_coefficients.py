import pytest
import torch

from cooper import multipliers
from tests.helpers import testing_utils


@pytest.fixture(params=[multipliers.DensePenaltyCoefficient, multipliers.IndexedPenaltyCoefficient])
def penalty_coefficient_class(request):
    return request.param


@pytest.fixture(params=[1, 100])
def num_constraints(request):
    return request.param


def evaluate_penalty_coefficient(penalty_coefficient, indices):
    if isinstance(penalty_coefficient, multipliers.IndexedPenaltyCoefficient):
        return penalty_coefficient(indices)
    else:
        # Ignore the indices for non-indexed penalty coefficients
        return penalty_coefficient()


def test_penalty_coefficient_init_and_forward(penalty_coefficient_class, num_constraints):
    generator = testing_utils.frozen_rand_generator()
    init_tensor = torch.randn(num_constraints, generator=generator)
    penalty_coefficient = penalty_coefficient_class(init_tensor)
    indices = torch.arange(num_constraints, dtype=torch.long)

    assert torch.equal(evaluate_penalty_coefficient(penalty_coefficient, indices), init_tensor)


def test_penalty_coefficient_failure_with_grad(penalty_coefficient_class, num_constraints):
    generator = testing_utils.frozen_rand_generator()
    with pytest.raises(ValueError, match="PenaltyCoefficient should not require gradients."):
        penalty_coefficient_class(torch.randn(num_constraints, requires_grad=True, generator=generator))


def test_penalty_coefficient_failure_with_wrong_shape(penalty_coefficient_class):
    generator = testing_utils.frozen_rand_generator()
    with pytest.raises(ValueError, match="init must either be a scalar or a 1D tensor"):
        penalty_coefficient_class(torch.randn(1, generator=generator).unsqueeze(0))


def test_indexed_penalty_coefficient_forward_invalid_indices(num_constraints):
    multiplier = multipliers.IndexedPenaltyCoefficient(torch.randn(num_constraints))
    indices = torch.arange(num_constraints, dtype=torch.float32)

    with pytest.raises(ValueError, match="Indices must be of type torch.long."):
        multiplier(indices)


def test_save_and_load_state_dict(penalty_coefficient_class, num_constraints):
    generator = testing_utils.frozen_rand_generator()
    init_tensor = torch.randn(num_constraints, generator=generator)

    penalty_coefficient = penalty_coefficient_class(init_tensor)
    indices = torch.arange(num_constraints, dtype=torch.long)
    penalty_coefficient_value = evaluate_penalty_coefficient(penalty_coefficient, indices)
    state_dict = penalty_coefficient.state_dict()

    new_penalty_coefficient = penalty_coefficient_class(torch.randn(num_constraints, generator=generator))
    new_penalty_coefficient.load_state_dict(state_dict)
    new_penalty_coefficient_value = evaluate_penalty_coefficient(new_penalty_coefficient, indices)

    assert torch.equal(new_penalty_coefficient_value, penalty_coefficient_value)
