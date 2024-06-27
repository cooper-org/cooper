import pytest
import torch

from tests.helpers import cooper_test_utils

PRIMAL_LR = 3e-2


class TestConvergenceNoConstraint:
    @pytest.fixture(autouse=True)
    def setup_cmp(self, num_variables, device):
        self.cmp = cooper_test_utils.SquaredNormLinearCMP(num_variables=num_variables, device=device)
        self.num_variables = num_variables
        self.device = device

    def test_convergence_no_constraint(self, use_multiple_primal_optimizers):
        x_init = torch.ones(self.num_variables, device=self.device)
        x_init = x_init.tensor_split(2) if use_multiple_primal_optimizers else [x_init]
        params = list(map(lambda t: torch.nn.Parameter(t), x_init))
        primal_optimizers = cooper_test_utils.build_primal_optimizers(
            params, primal_optimizer_kwargs=[{"lr": PRIMAL_LR} for _ in range(len(params))]
        )

        cooper_optimizer = cooper_test_utils.build_cooper_optimizer(cmp=self.cmp, primal_optimizers=primal_optimizers)

        for _ in range(2000):
            cooper_optimizer.roll(compute_cmp_state_kwargs=dict(x=torch.cat(params)))

        # Compute the exact solution
        x_star, lambda_star = self.cmp.compute_exact_solution()

        # Check if the primal variable is close to the exact solution
        assert torch.allclose(torch.cat(params), x_star, atol=1e-5)
