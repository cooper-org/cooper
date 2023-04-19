import multiplier_test_utils
import pytest
import torch

from cooper import multipliers


@pytest.fixture(params=[multipliers.DenseMultiplier, multipliers.SparseMultiplier])
def mult_class(request):
    return request.param


def test_eq_multiplier_init_and_forward(mult_class, init_tensor, all_indices):
    eq_multiplier = mult_class(init_tensor, restart_on_feasible=False)
    assert eq_multiplier.implicit_constraint_type == "eq"

    is_sparse = isinstance(eq_multiplier, multipliers.SparseMultiplier)
    multiplier_values = eq_multiplier(all_indices) if is_sparse else eq_multiplier()
    assert torch.allclose(multiplier_values, init_tensor)


def test_eq_post_step_(mult_class, init_tensor, all_indices, feasible_indices):
    eq_multiplier = mult_class(init_tensor, restart_on_feasible=False)
    eq_multiplier.post_step_(feasible_indices=feasible_indices)

    is_sparse = isinstance(eq_multiplier, multipliers.SparseMultiplier)
    multiplier_values = eq_multiplier(all_indices) if is_sparse else eq_multiplier()

    # Post step is a no-op for multipliers for equality constraints (no projection)
    assert torch.allclose(multiplier_values, init_tensor)


def test_save_load_multipliers(mult_class, init_tensor, all_indices, multiplier_shape, random_seed):
    eq_multiplier = mult_class(init_tensor, restart_on_feasible=False)
    multiplier_test_utils.check_save_load_state_dict(
        eq_multiplier, mult_class, multiplier_shape, all_indices, random_seed
    )


def test_sparse_multipliers():
    # TODO(juan43ramirez): implement a dedicated test for sparse multipliers
    pass
