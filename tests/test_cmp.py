import cooper_test_utils
import pytest
import testing_utils
import torch

import cooper


def build_optimizer(params, cmp, use_ineq_constraints):

    primal_optimizer = torch.optim.SGD([params], lr=1e-2)

    if use_ineq_constraints:
        dual_params = [{"params": constraint.multiplier.parameters()} for constraint in cmp.constraint_groups]
        dual_optimizer = torch.optim.SGD(dual_params, lr=1e-2)

        constrained_optimizer = cooper.optim.SimultaneousConstrainedOptimizer(
            primal_optimizers=primal_optimizer, dual_optimizers=dual_optimizer, constraint_groups=cmp.constraint_groups
        )

        return constrained_optimizer
    else:
        unconstrained_optimizer = cooper.optim.UnconstrainedOptimizer(primal_optimizers=primal_optimizer)

        return unconstrained_optimizer


def test_cmp(device, Toy2dCMP_params_init, Toy2dCMP_problem_properties):

    use_ineq_constraints = Toy2dCMP_problem_properties["use_ineq_constraints"]
    solution = Toy2dCMP_problem_properties["solution"]

    params = torch.nn.Parameter(Toy2dCMP_params_init)
    cmp = cooper_test_utils.Toy2dCMP(use_ineq_constraints=use_ineq_constraints, device=device)

    cooper_optimizer = build_optimizer(params, cmp, use_ineq_constraints)

    for step_id in range(1500):
        cooper_optimizer.zero_grad()
        cmp_state = cmp.compute_cmp_state(params)
        lagrangian = cmp_state.populate_lagrangian()
        cmp_state.backward()
        cooper_optimizer.step()

    assert torch.allclose(params, solution)
