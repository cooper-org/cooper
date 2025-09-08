# Copyright (C) 2025 The Cooper Developers.
# Licensed under the MIT License.

import os
import tempfile

import pytest
import torch

import cooper
import testing


def evaluate_multiplier(multiplier, all_indices):
    """Helper function for consistently evaluating Indexed/Explicit multipliers."""
    return multiplier(all_indices) if multiplier.expects_constraint_features else multiplier()


def test_multiplier_initialization_with_init(multiplier, init_multiplier_tensor, device):
    assert torch.equal(multiplier.weight.view(-1), init_multiplier_tensor.to(device).view(-1))
    assert multiplier.device.type == device.type


def test_multiplier_initialization_with_num_constraints(multiplier, num_constraints, device):
    assert multiplier.weight.numel() == num_constraints
    assert multiplier.device.type == device.type


def test_multiplier_initialization_without_init_or_num_constraints(multiplier_class):
    with pytest.raises(ValueError, match=r"At least one of `num_constraints` and `init` must be provided."):
        multiplier_class()


def test_multiplier_initialization_with_inconsistent_init_shape(multiplier_class, num_constraints):
    with pytest.raises(ValueError, match="Inconsistent `init` shape"):
        multiplier_class(num_constraints=num_constraints, init=torch.zeros(num_constraints + 1))


def test_multiplier_initialization_with_init_dim(multiplier_class, num_constraints):
    with pytest.raises(ValueError, match=r"`init` must be a 1D tensor of shape"):
        multiplier_class(num_constraints=num_constraints, init=torch.zeros(num_constraints, 1))


def test_multiplier_repr(multiplier, multiplier_class, num_constraints):
    assert repr(multiplier) == f"{multiplier_class.__name__}(num_constraints={num_constraints})"


def test_multiplier_sanity_check(constraint_type, multiplier_class, init_multiplier_tensor):
    # Force the initialization tensor to have negative entries
    if constraint_type == cooper.ConstraintType.EQUALITY:
        pytest.skip("")

    neg_multiplier = multiplier_class(init=init_multiplier_tensor.abs().neg())
    with pytest.raises(ValueError, match=r"For inequality constraint, all entries in multiplier must be non-negative."):
        neg_multiplier.set_constraint_type(cooper.ConstraintType.INEQUALITY)


def test_multiplier_init_and_forward(multiplier, init_multiplier_tensor, all_indices):
    # Ensure that the multiplier returns the correct value when called
    multiplier_values = evaluate_multiplier(multiplier, all_indices)
    target_tensor = init_multiplier_tensor.reshape(multiplier_values.shape)
    assert torch.allclose(multiplier_values, target_tensor)


def test_equality_post_step_(constraint_type, multiplier, init_multiplier_tensor, all_indices):
    """Post-step for equality multipliers should be a no-op. Check that multiplier
    values remain unchanged after calling post_step_.
    """
    if constraint_type == cooper.ConstraintType.INEQUALITY:
        pytest.skip("")

    multiplier.set_constraint_type(constraint_type)
    multiplier.post_step_()
    multiplier_values = evaluate_multiplier(multiplier, all_indices)
    target_tensor = init_multiplier_tensor.reshape(multiplier_values.shape)
    assert torch.allclose(multiplier_values, target_tensor)


def test_ineq_post_step_(constraint_type, multiplier, all_indices):
    """Ensure that the inequality multipliers remain non-negative after post-step."""
    if constraint_type == cooper.ConstraintType.EQUALITY:
        pytest.skip("")

    multiplier.set_constraint_type(constraint_type)

    # Overwrite the multiplier to have some *negative* entries and gradients
    hard_coded_weight_data = torch.randn_like(multiplier.weight)
    multiplier.weight.data = hard_coded_weight_data

    hard_coded_gradient_data = torch.randn_like(multiplier.weight)
    if isinstance(multiplier, cooper.multipliers.IndexedMultiplier) and multiplier.sparse_grad:
        hard_coded_gradient_data = hard_coded_gradient_data.to_sparse(sparse_dim=1)
    multiplier.weight.grad = hard_coded_gradient_data

    # Post-step should ensure non-negativity
    multiplier.post_step_()
    multiplier_values = evaluate_multiplier(multiplier, all_indices)

    target_weight_data = hard_coded_weight_data.relu()
    current_grad = multiplier.weight.grad

    assert torch.allclose(multiplier_values, target_weight_data)
    assert torch.allclose(current_grad.to_dense(), hard_coded_gradient_data.to_dense())


def check_save_load_state_dict(multiplier, explicit_multiplier_class, num_constraints, random_seed):
    generator = testing.frozen_rand_generator(random_seed)

    multiplier_init = torch.randn(num_constraints, generator=generator)
    new_multiplier = explicit_multiplier_class(init=multiplier_init)

    # Save to file to force reading from file so we can ensure correct loading
    with tempfile.TemporaryDirectory() as tmpdirname:
        torch.save(multiplier.state_dict(), os.path.join(tmpdirname, "multiplier.pt"))
        state_dict = torch.load(os.path.join(tmpdirname, "multiplier.pt"), weights_only=True)

    new_multiplier.load_state_dict(state_dict)

    assert torch.equal(multiplier.weight, new_multiplier.weight)


def test_save_load_multiplier(multiplier, multiplier_class, num_constraints, random_seed):
    """Test that the state_dict of a multiplier can be saved and loaded correctly."""
    check_save_load_state_dict(multiplier, multiplier_class, num_constraints, random_seed)


def test_multiplier_grad(multiplier, all_indices):
    evaluate_multiplier(multiplier, all_indices).sum().backward()
    assert multiplier.weight.grad.is_sparse == (
        isinstance(multiplier, cooper.multipliers.IndexedMultiplier) and multiplier.sparse_grad
    )
