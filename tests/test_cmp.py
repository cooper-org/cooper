import cooper_test_utils
import pytest
import testing_utils
import torch

import cooper


def test_cmp(device):

    params = torch.nn.Parameter(torch.tensor([0.1, 0.5], device=device))
    cmp = cooper_test_utils.Toy2dCMP(use_ineq_constraints=True, device=device)

    primal_optimizer = torch.optim.SGD([params], lr=1e-2)

    dual_params = [{"params": constraint.multiplier.parameters()} for constraint in cmp.constraint_groups]
    dual_optimizer = torch.optim.SGD(dual_params, lr=1e-2)

    constrained_optimizer = cooper.optim.SimultaneousConstrainedOptimizer(
        primal_optimizers=primal_optimizer, dual_optimizers=dual_optimizer, constraint_groups=cmp.constraint_groups
    )

    for step_id in range(1000):

        constrained_optimizer.zero_grad()
        cmp_state = cmp.compute_cmp_state(params)
        lagrangian, multipliers = cmp_state.populate_lagrangian(return_multipliers=True)
        cmp_state.backward()
        constrained_optimizer.step()

    # Constrained optimum is located at [2/3, 1/3]
    assert torch.allclose(params[0], torch.tensor(2.0 / 3.0))
    assert torch.allclose(params[1], torch.tensor(1.0 / 3.0))
