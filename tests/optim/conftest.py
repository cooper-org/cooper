import pytest

import cooper


@pytest.fixture
def cmp_state():
    return cooper.CMPState()


@pytest.fixture
def cmp_instance(cmp_state):
    class DummyCMP(cooper.ConstrainedMinimizationProblem):
        def __init__(self):
            super().__init__()
            self.eq_constraint = cooper.Constraint(
                constraint_type=cooper.ConstraintType.EQUALITY,
                multiplier=cooper.multipliers.DenseMultiplier(num_constraints=1),
            )

        def compute_cmp_state(self):  # noqa: PLR6301
            return cmp_state

    return DummyCMP()
