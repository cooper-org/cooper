import pytest
import torch

import cooper
from cooper.constraints import ConstraintState, ConstraintType


def test_penalty_coefficient_check():
    # Create an instance of AugmentedLagrangianFormulation
    formulation = cooper.AugmentedLagrangianFormulation(constraint_type=ConstraintType.EQUALITY)

    # Create instances of ConstraintState and Multiplier
    constraint_state = ConstraintState(violation=torch.tensor([1.0]))
    multiplier = cooper.multipliers.DenseMultiplier(constraint_type=ConstraintType.EQUALITY, num_constraints=1)

    # Call compute_contribution_to_primal_lagrangian with penalty_coefficient set to None
    # Expect a ValueError to be raised
    with pytest.raises(ValueError, match=r".*expects a penalty coefficient but none was provided.*"):
        formulation.compute_contribution_to_primal_lagrangian(
            constraint_state=constraint_state, multiplier=multiplier, penalty_coefficient=None
        )
