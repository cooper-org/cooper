from dataclasses import is_dataclass

import torch

import cooper


# Dummy Constraint class to use as key in dictionaries
class Constraint:
    pass


def test_lagrangian_store_initialization():
    lagrangian_store = cooper.LagrangianStore()
    assert lagrangian_store.lagrangian is None
    assert isinstance(lagrangian_store.multiplier_values, dict)
    assert isinstance(lagrangian_store.penalty_coefficient_values, dict)


def test_lagrangian_store_dataclass():
    assert is_dataclass(cooper.LagrangianStore)


def test_lagrangian_store_backward():
    lagrangian_store = cooper.LagrangianStore(lagrangian=torch.tensor(1.0, requires_grad=True))
    lagrangian_store.backward()
    assert lagrangian_store.lagrangian.grad is not None


def test_lagrangian_store_backward_none():
    lagrangian_store = cooper.LagrangianStore()
    lagrangian_store.backward()
    assert True  # No error should be raised


def test_lagrangian_store_observed_multiplier_values():
    constraint1 = Constraint()
    constraint2 = Constraint()
    lagrangian_store = cooper.LagrangianStore(
        multiplier_values={constraint1: torch.tensor(1.0), constraint2: torch.tensor(2.0)}
    )
    observed_values = list(lagrangian_store.observed_multiplier_values())
    assert observed_values == [torch.tensor(1.0), torch.tensor(2.0)]


def test_lagrangian_store_observed_penalty_coefficient_values():
    constraint1 = Constraint()
    constraint2 = Constraint()
    lagrangian_store = cooper.LagrangianStore(
        penalty_coefficient_values={constraint1: torch.tensor(3.0), constraint2: torch.tensor(4.0)}
    )
    observed_values = list(lagrangian_store.observed_penalty_coefficient_values())
    assert observed_values == [torch.tensor(3.0), torch.tensor(4.0)]
