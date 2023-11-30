import multiplier_test_utils
import pytest
import torch

import cooper


@pytest.fixture(params=[True, False])
def restart_on_feasible(request):
    return request.param


CONSTRAINT_TYPE = cooper.ConstraintType.INEQUALITY


def test_ineq_multiplier_init_and_forward(multiplier_class, restart_on_feasible, init_tensor, all_indices):
    ineq_multiplier = multiplier_class(
        constraint_type=CONSTRAINT_TYPE, init=init_tensor.relu(), restart_on_feasible=restart_on_feasible
    )

    is_indexed = isinstance(ineq_multiplier, cooper.multipliers.IndexedMultiplier)
    multiplier_values = ineq_multiplier(all_indices) if is_indexed else ineq_multiplier()
    # The multiplier is initialized with a non-negative version of `init_tensor`
    target_tensor = init_tensor.relu().reshape(multiplier_values.shape)

    assert torch.allclose(multiplier_values, target_tensor)


def test_ineq_post_step_(multiplier_class, restart_on_feasible, init_tensor, feasible_indices, all_indices):
    ineq_multiplier = multiplier_class(
        constraint_type=CONSTRAINT_TYPE, init=init_tensor.relu(), restart_on_feasible=restart_on_feasible
    )
    is_indexed = isinstance(ineq_multiplier, cooper.multipliers.IndexedMultiplier)

    # Overwrite the multiplier to have some *negative* entries and gradients
    hard_coded_weight_data = torch.randn_like(ineq_multiplier.weight)
    ineq_multiplier.weight.data = hard_coded_weight_data

    hard_coded_gradient_data = torch.randn_like(ineq_multiplier.weight)
    ineq_multiplier.weight.grad = hard_coded_gradient_data
    if isinstance(ineq_multiplier, cooper.multipliers.IndexedMultiplier):
        ineq_multiplier.weight.grad = ineq_multiplier.weight.grad.to_sparse(sparse_dim=1)

    # Post-step should ensure non-negativity. Note that no feasible indices are passed,
    # so "feasible" multipliers and their gradients are not reset.
    ineq_multiplier.post_step_()
    multiplier_values = ineq_multiplier(all_indices) if is_indexed else ineq_multiplier()

    target_weight_data = hard_coded_weight_data.relu().reshape_as(multiplier_values)
    current_grad = ineq_multiplier.weight.grad.to_dense()
    assert torch.allclose(multiplier_values, target_weight_data)
    assert torch.allclose(current_grad, hard_coded_gradient_data)

    # Perform post-step again, this time with feasible indices
    ineq_multiplier.strictly_feasible_indices = feasible_indices
    ineq_multiplier.post_step_()

    multiplier_values = ineq_multiplier(all_indices) if is_indexed else ineq_multiplier()

    current_grad = ineq_multiplier.weight.grad.to_dense()
    if not ineq_multiplier.restart_on_feasible:
        # Latest post-step is a no-op
        assert torch.allclose(multiplier_values, target_weight_data)
    else:
        assert torch.allclose(multiplier_values[feasible_indices], torch.tensor(0.0))
        assert torch.allclose(multiplier_values[~feasible_indices], target_weight_data[~feasible_indices])

    assert torch.allclose(current_grad, hard_coded_gradient_data)


def test_save_load_multiplier(
    multiplier_class, restart_on_feasible, init_tensor, multiplier_shape, all_indices, random_seed
):
    ineq_multiplier = multiplier_class(
        constraint_type=CONSTRAINT_TYPE, init=init_tensor.relu(), restart_on_feasible=restart_on_feasible
    )
    multiplier_test_utils.check_save_load_state_dict(
        ineq_multiplier, multiplier_class, multiplier_shape, all_indices, random_seed
    )


def test_indexed_ineq_multipliers():
    # TODO(juan43ramirez): implement a dedicated test for indexed multipliers
    pass
