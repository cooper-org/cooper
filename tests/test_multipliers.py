#!/usr/bin/env python

"""Tests for Multiplier class."""

import tempfile

import pytest
import testing_utils
import torch

from cooper import multipliers


@pytest.fixture
def random_seed():
    return 1516516984916


@pytest.fixture(params=[multipliers.DenseMultiplier, multipliers.SparseMultiplier])
def explicit_multiplier_class(request):
    return request.param


@pytest.fixture
def multiplier_shape():
    return (100, 1)


@pytest.fixture
def init_tensor(multiplier_shape, random_seed):
    generator = testing_utils.frozen_rand_generator(random_seed)
    return torch.randn(*multiplier_shape, generator=generator)


@pytest.fixture
def all_indices(multiplier_shape):
    return torch.arange(0, multiplier_shape[0], dtype=torch.long)


@pytest.fixture
def eq_multiplier(explicit_multiplier_class, init_tensor):
    return explicit_multiplier_class(init_tensor, restart_on_feasible=False)


@pytest.fixture(params=[True, False])
def ineq_multiplier(explicit_multiplier_class, init_tensor, request):
    restart_on_feasible = request.param
    return explicit_multiplier_class(init_tensor.relu(), enforce_positive=True, restart_on_feasible=restart_on_feasible)


@pytest.fixture
def feasible_indices(multiplier_shape, random_seed):
    generator = testing_utils.frozen_rand_generator(random_seed)
    return torch.randint(0, 2, multiplier_shape, dtype=torch.bool, generator=generator)


def test_constant_multiplier_init_and_forward(init_tensor):
    multiplier = multipliers.ConstantMultiplier(init_tensor)
    assert torch.allclose(multiplier(), init_tensor)


def test_eq_multiplier_init_and_forward(eq_multiplier, init_tensor, all_indices):

    assert eq_multiplier.implicit_constraint_type == "eq"

    is_sparse = isinstance(eq_multiplier, multipliers.SparseMultiplier)
    multiplier_values = eq_multiplier(all_indices) if is_sparse else eq_multiplier()
    assert torch.allclose(multiplier_values, init_tensor)


def test_ineq_multiplier_init_and_forward(ineq_multiplier, init_tensor, all_indices):

    assert ineq_multiplier.implicit_constraint_type == "ineq"

    is_sparse = isinstance(ineq_multiplier, multipliers.SparseMultiplier)
    multiplier_values = ineq_multiplier(all_indices) if is_sparse else ineq_multiplier()
    # The multiplier is initialized with a non-negative projected version of `init_tensor`
    assert torch.allclose(multiplier_values, init_tensor.relu())


def test_eq_post_step_(eq_multiplier, feasible_indices, init_tensor, all_indices):

    eq_multiplier.post_step_(feasible_indices=feasible_indices)

    is_sparse = isinstance(eq_multiplier, multipliers.SparseMultiplier)
    multiplier_values = eq_multiplier(all_indices) if is_sparse else eq_multiplier()

    # Post step is a no-op for multipliers for equality constraints (no projection)
    assert torch.allclose(multiplier_values, init_tensor)


def test_ineq_post_step_(ineq_multiplier, feasible_indices, init_tensor, all_indices):

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


def check_save_load_state_dict(multiplier, explicit_multiplier_class, multiplier_shape, all_indices, random_seed):

    generator = testing_utils.frozen_rand_generator(random_seed)
    new_multiplier = explicit_multiplier_class(init=torch.randn(*multiplier_shape, generator=generator))

    # Save to file to force reading from file so we can ensure correct loading
    tmp = tempfile.NamedTemporaryFile()
    torch.save(multiplier.state_dict(), tmp.name)
    state_dict = torch.load(tmp.name)
    tmp.close()

    new_multiplier.load_state_dict(state_dict)

    assert multiplier.implicit_constraint_type == new_multiplier.implicit_constraint_type
    assert multiplier.restart_on_feasible == new_multiplier.restart_on_feasible

    is_sparse = isinstance(multiplier, multipliers.SparseMultiplier)
    if is_sparse:
        assert torch.allclose(multiplier(all_indices), new_multiplier(all_indices))
    else:
        assert torch.allclose(multiplier(), new_multiplier())


def test_save_load_multipliers(
    ineq_multiplier, eq_multiplier, explicit_multiplier_class, multiplier_shape, all_indices, random_seed
):
    check_save_load_state_dict(eq_multiplier, explicit_multiplier_class, multiplier_shape, all_indices, random_seed)
    check_save_load_state_dict(ineq_multiplier, explicit_multiplier_class, multiplier_shape, all_indices, random_seed)


def test_sparse_multipliers():
    # TODO(juan43ramirez): implement a dedicated test for sparse multipliers
    pass
