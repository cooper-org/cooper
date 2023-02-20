#!/usr/bin/env python

"""Tests for Multiplier class."""

import tempfile

import pytest
import torch

from cooper import multipliers


@pytest.fixture(params=[multipliers.DenseMultiplier, multipliers.SparseMultiplier])
def explicit_multiplier_class(request):
    return request.param


@pytest.fixture
def is_sparse(explicit_multiplier_class):
    return explicit_multiplier_class == multipliers.SparseMultiplier


@pytest.fixture
def multiplier_shape():
    return (100, 1)


@pytest.fixture
def init_tensor(multiplier_shape):
    return torch.randn(*multiplier_shape)


@pytest.fixture(params=[True, False])
def restart_on_feasible(request):
    return request.param


@pytest.fixture
def all_indices(multiplier_shape):
    return torch.arange(0, multiplier_shape[0], dtype=torch.long)


@pytest.fixture
def eq_multiplier(explicit_multiplier_class, init_tensor, restart_on_feasible):
    return explicit_multiplier_class(init_tensor, restart_on_feasible=restart_on_feasible)


@pytest.fixture
def ineq_multiplier(explicit_multiplier_class, init_tensor, restart_on_feasible):
    return explicit_multiplier_class(init_tensor.relu(), enforce_positive=True, restart_on_feasible=restart_on_feasible)


@pytest.fixture
def feasible_indices(multiplier_shape):
    return torch.randint(0, 2, multiplier_shape, dtype=torch.bool)


def test_constant_multiplier_init(init_tensor):
    multiplier = multipliers.ConstantMultiplier(init_tensor)
    assert torch.allclose(multiplier(), init_tensor)


def test_eq_multiplier_init(eq_multiplier, init_tensor, all_indices, is_sparse):

    # if is_sparse:
    #     breakpoint()

    multiplier_values = eq_multiplier(all_indices) if is_sparse else eq_multiplier()

    assert torch.allclose(multiplier_values, init_tensor)
    assert eq_multiplier.implicit_constraint_type == "eq"


def test_ineq_multiplier_init(ineq_multiplier, init_tensor, all_indices, is_sparse):

    multiplier_values = ineq_multiplier(all_indices) if is_sparse else ineq_multiplier()

    assert torch.allclose(multiplier_values, init_tensor.relu())
    assert ineq_multiplier.implicit_constraint_type == "ineq"


def test_eq_post_step(eq_multiplier, feasible_indices, init_tensor, restart_on_feasible, all_indices, is_sparse):

    eq_multiplier.post_step(feasible_indices=feasible_indices)
    multiplier_values = eq_multiplier(all_indices) if is_sparse else eq_multiplier()

    if not restart_on_feasible:
        # Post step is a no-op
        assert torch.allclose(multiplier_values, init_tensor)
    else:
        # Multipliers at feasible indices are restarted (by default to 0.0)
        assert torch.allclose(multiplier_values[feasible_indices], torch.tensor(0.0))
        assert torch.allclose(multiplier_values[~feasible_indices], init_tensor[~feasible_indices])


def test_ineq_post_step(ineq_multiplier, feasible_indices, init_tensor, restart_on_feasible, all_indices, is_sparse):

    # Overwrite the multiplier to have some *negative* entries and gradients
    ineq_multiplier.weight.data = init_tensor.clone()
    ineq_multiplier.weight.grad = init_tensor.clone()

    # Post-step should ensure non-negativity. Note that no feasible indices are passed,
    # so "feasible" multipliers and their gradients are not reset.
    ineq_multiplier.post_step()
    multiplier_values = ineq_multiplier(all_indices) if is_sparse else ineq_multiplier()

    assert torch.allclose(multiplier_values, init_tensor.relu())
    assert torch.allclose(ineq_multiplier.weight.grad, init_tensor)

    # Perform post-step again, this time with feasible indices
    ineq_multiplier.post_step(feasible_indices=feasible_indices)
    multiplier_values = ineq_multiplier(all_indices) if is_sparse else ineq_multiplier()

    if not restart_on_feasible:
        # Latest post-step is a no-op
        assert torch.allclose(multiplier_values, init_tensor.relu())
        assert torch.allclose(ineq_multiplier.weight.grad, init_tensor)
    else:
        assert torch.allclose(multiplier_values[feasible_indices], torch.tensor(0.0))
        assert torch.allclose(ineq_multiplier.weight.grad[feasible_indices], torch.tensor(0.0))

        assert torch.allclose(multiplier_values[~feasible_indices], init_tensor.relu()[~feasible_indices])
        assert torch.allclose(ineq_multiplier.weight.grad[~feasible_indices], init_tensor[~feasible_indices])


def test_save_load_state_dict(init_tensor, explicit_multiplier_class, restart_on_feasible, all_indices, is_sparse):

    multiplier = explicit_multiplier_class(init_tensor, restart_on_feasible=restart_on_feasible)
    new_multiplier = explicit_multiplier_class(torch.randn(100, 1))

    tmp = tempfile.NamedTemporaryFile()
    torch.save(multiplier.state_dict(), tmp.name)
    state_dict = torch.load(tmp.name)
    tmp.close()

    new_multiplier.load_state_dict(state_dict)

    if is_sparse:
        assert torch.allclose(multiplier(all_indices), new_multiplier(all_indices))
    else:
        assert torch.allclose(multiplier(), new_multiplier())
    assert multiplier.implicit_constraint_type == new_multiplier.implicit_constraint_type
    assert multiplier.restart_on_feasible == new_multiplier.restart_on_feasible


def test_sparse_multipliers():
    # TODO(juan43ramirez): implement a dedicated test for sparse multipliers
    pass
