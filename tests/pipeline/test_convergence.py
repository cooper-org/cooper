import pytest
import torch

from tests.helpers import cooper_test_utils

PRIMAL_LR = 3e-2


@pytest.fixture
def cooper_optimizer_no_constraint(cmp_no_constraint, params):
    primal_optimizers = cooper_test_utils.build_primal_optimizers(
        params, primal_optimizer_kwargs=[{"lr": PRIMAL_LR} for _ in range(len(params))]
    )
    cooper_optimizer = cooper_test_utils.build_cooper_optimizer(
        cmp=cmp_no_constraint, primal_optimizers=primal_optimizers
    )
    return cooper_optimizer


def test_convergence_no_constraint(cmp_no_constraint, params, cooper_optimizer_no_constraint):
    for _ in range(2000):
        cooper_optimizer_no_constraint.roll(compute_cmp_state_kwargs=dict(x=torch.cat(params)))

    # Compute the exact solution
    x_star, _ = cmp_no_constraint.compute_exact_solution()

    # Check if the primal variable is close to the exact solution
    assert torch.allclose(torch.cat(params), x_star, atol=1e-5)
