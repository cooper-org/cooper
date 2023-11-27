import multiplier_test_utils
import pytest
import torch

import cooper

CONSTRAINT_TYPE = cooper.ConstraintType.EQUALITY


def test_eq_multiplier_init_and_forward(multiplier_class, init_tensor, all_indices):
    eq_multiplier = multiplier_class(constraint_type=CONSTRAINT_TYPE, init=init_tensor)

    is_indexed = isinstance(eq_multiplier, cooper.multipliers.IndexedMultiplier)
    multiplier_values = eq_multiplier(all_indices) if is_indexed else eq_multiplier()

    assert torch.allclose(multiplier_values, init_tensor.reshape_as(multiplier_values))


def test_eq_post_step_(multiplier_class, init_tensor, all_indices, feasible_indices):
    eq_multiplier = multiplier_class(constraint_type=CONSTRAINT_TYPE, init=init_tensor)
    eq_multiplier.strictly_feasible_indices = feasible_indices
    eq_multiplier.post_step_()

    is_indexed = isinstance(eq_multiplier, cooper.multipliers.IndexedMultiplier)
    multiplier_values = eq_multiplier(all_indices) if is_indexed else eq_multiplier()

    # Post step is a no-op for multipliers for equality constraints (no projection)
    assert torch.allclose(multiplier_values, init_tensor.reshape_as(multiplier_values))


def test_save_load_multipliers(multiplier_class, init_tensor, all_indices, multiplier_shape, random_seed):
    eq_multiplier = multiplier_class(constraint_type=CONSTRAINT_TYPE, init=init_tensor)
    multiplier_test_utils.check_save_load_state_dict(
        eq_multiplier, multiplier_class, multiplier_shape, all_indices, random_seed
    )


def test_indexed_eq_multipliers():
    # TODO(juan43ramirez): implement a dedicated test for indexed multipliers
    pass
