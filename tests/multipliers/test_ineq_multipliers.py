import multiplier_test_utils
import pytest
import torch

from cooper import multipliers


@pytest.fixture(params=[multipliers.DenseMultiplier, multipliers.SparseMultiplier])
def mult_class(request):
    return request.param


@pytest.fixture(params=[True, False])
def restart_on_feasible(request):
    return request.param


def test_ineq_multiplier_init_and_forward(mult_class, restart_on_feasible, init_tensor, all_indices):
    ineq_multiplier = mult_class(init_tensor.relu(), enforce_positive=True, restart_on_feasible=restart_on_feasible)
    assert ineq_multiplier.implicit_constraint_type == "ineq"

    is_sparse = isinstance(ineq_multiplier, multipliers.SparseMultiplier)
    multiplier_values = ineq_multiplier(all_indices) if is_sparse else ineq_multiplier()
    # The multiplier is initialized with a non-negative version of `init_tensor`
    assert torch.allclose(multiplier_values, init_tensor.relu())


def test_ineq_post_step_(mult_class, restart_on_feasible, init_tensor, feasible_indices, all_indices):
    ineq_multiplier = mult_class(init_tensor.relu(), enforce_positive=True, restart_on_feasible=restart_on_feasible)
    is_sparse = isinstance(ineq_multiplier, multipliers.SparseMultiplier)

    # Overwrite the multiplier to have some *negative* entries and gradients
    ineq_multiplier.weight.data = init_tensor.clone()
    ineq_multiplier.weight.grad = init_tensor.clone()

    # Post-step should ensure non-negativity. Note that no feasible indices are passed,
    # so "feasible" multipliers and their gradients are not reset.
    ineq_multiplier.post_step_()
    multiplier_values = ineq_multiplier(all_indices) if is_sparse else ineq_multiplier()

    assert torch.allclose(multiplier_values, init_tensor.relu())
    assert torch.allclose(ineq_multiplier.weight.grad, init_tensor)

    # Perform post-step again, this time with feasible indices
    ineq_multiplier.post_step_(feasible_indices=feasible_indices)

    multiplier_values = ineq_multiplier(all_indices) if is_sparse else ineq_multiplier()

    if not ineq_multiplier.restart_on_feasible:
        # Latest post-step is a no-op
        assert torch.allclose(multiplier_values, init_tensor.relu())
        assert torch.allclose(ineq_multiplier.weight.grad, init_tensor)
    else:
        assert torch.allclose(multiplier_values[feasible_indices], torch.tensor(0.0))
        assert torch.allclose(ineq_multiplier.weight.grad[feasible_indices], torch.tensor(0.0))

        assert torch.allclose(multiplier_values[~feasible_indices], init_tensor.relu()[~feasible_indices])
        assert torch.allclose(ineq_multiplier.weight.grad[~feasible_indices], init_tensor[~feasible_indices])


def test_save_load_multiplier(mult_class, restart_on_feasible, init_tensor, multiplier_shape, all_indices, random_seed):
    ineq_multiplier = mult_class(init_tensor.relu(), enforce_positive=True, restart_on_feasible=restart_on_feasible)
    multiplier_test_utils.check_save_load_state_dict(
        ineq_multiplier, mult_class, multiplier_shape, all_indices, random_seed
    )


def test_sparse_multipliers():
    # TODO(juan43ramirez): implement a dedicated test for sparse multipliers
    pass
