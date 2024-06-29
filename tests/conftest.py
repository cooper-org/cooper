import pytest
import torch

import cooper


@pytest.fixture(scope="session", params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("Aim device 'cuda' is not available.")
    return torch.device(request.param)


@pytest.fixture
def eq_constraint():
    constraint = cooper.Constraint(
        constraint_type=cooper.ConstraintType.EQUALITY,
        multiplier=cooper.multipliers.DenseMultiplier(
            constraint_type=cooper.ConstraintType.EQUALITY, num_constraints=1
        ),
    )
    return constraint


@pytest.fixture
def ineq_constraint():
    constraint = cooper.Constraint(
        constraint_type=cooper.ConstraintType.INEQUALITY,
        multiplier=cooper.multipliers.DenseMultiplier(
            constraint_type=cooper.ConstraintType.INEQUALITY, num_constraints=1
        ),
    )
    return constraint


@pytest.fixture
def cmp_state():
    return cooper.CMPState()


class DummyCMP(cooper.ConstrainedMinimizationProblem):
    def compute_cmp_state(self):
        return cmp_state


@pytest.fixture
def cmp_instance():
    return DummyCMP()
