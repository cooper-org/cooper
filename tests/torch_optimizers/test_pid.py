import cooper_test_utils
import pytest
import torch

import cooper
from cooper.optim import PID, SparsePID


@pytest.mark.parametrize(["proportional", "integral", "derivative"], [(0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1)])
def test_manual_pid(proportional, integral, derivative):
    param = torch.tensor([1.0], requires_grad=True)

    loss_fn = lambda param: param**2 / 2

    lr = 0.1

    def PID_direction(grad, prev_grad, prev_change):
        change = grad - prev_grad
        curvature = change - prev_change

        return integral * grad + proportional * change + derivative * curvature

    optimizer = PID([param], lr=lr, proportional=proportional, integral=integral, derivative=derivative)

    def do_step():
        optimizer.zero_grad()
        loss = loss_fn(param)
        loss.backward()
        optimizer.step()

    # -------------------------------------------- First step of optimization

    # Manual first step of optimization. The gradient is simply the current value of p.
    p0 = param.clone().detach()
    p1 = p0 - lr * PID_direction(p0, 0.0, 0.0)

    do_step()

    assert torch.allclose(param, p1)

    # Check the state of the optimizer. Should contain the first gradient in
    # `previous_direction` and also in `previous_change`.
    assert torch.allclose(optimizer.state[param]["previous_direction"], p0)
    assert torch.allclose(optimizer.state[param]["previous_change"], p0)

    # -------------------------------------------- Second step of optimization
    p2 = p1 - lr * PID_direction(p1, p0, p0)

    do_step()

    assert torch.allclose(param, p2)

    # The state contain p1 in `previous_direction` and p1 - p0 in `previous_change`.
    assert torch.allclose(optimizer.state[param]["previous_direction"], p1)
    assert torch.allclose(optimizer.state[param]["previous_change"], p1 - p0)

    # -------------------------------------------- Third step of optimization
    p3 = p2 - lr * PID_direction(p2, p1, p1 - p0)

    do_step()

    assert torch.allclose(param, p3)

    # The state contain p2 in `previous_direction` and p2 - p1 in `previous_change`.
    assert torch.allclose(optimizer.state[param]["previous_direction"], p2)
    assert torch.allclose(optimizer.state[param]["previous_change"], p2 - p1)


@pytest.mark.parametrize(["proportional", "integral", "derivative"], [(0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1)])
def test_pid_convergence(
    proportional,
    integral,
    derivative,
    Toy2dCMP_problem_properties,
    Toy2dCMP_params_init,
    use_multiple_primal_optimizers,
    device,
):
    """Test convergence of PID updates on toy 2D problem. The PID updates are only
    applied to the dual variables."""

    use_ineq_constraints = Toy2dCMP_problem_properties["use_ineq_constraints"]
    if not use_ineq_constraints:
        pytest.skip("Test requires a problem with constraints.")

    cmp = cooper_test_utils.Toy2dCMP(use_ineq_constraints=use_ineq_constraints, device=device)

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=use_multiple_primal_optimizers, params_init=Toy2dCMP_params_init
    )

    dual_params = [{"params": constraint.multiplier.parameters()} for constraint in cmp.constraint_groups]
    dual_optimizer = PID(
        dual_params, lr=0.01, proportional=proportional, integral=integral, derivative=derivative, maximize=True
    )

    cooper_optimizer = cooper.optim.SimultaneousOptimizer(
        primal_optimizers, dual_optimizer, constraint_groups=cmp.constraint_groups
    )

    compute_cmp_state_fn = lambda: cmp.compute_cmp_state(params)

    for step_id in range(1500):
        cooper_optimizer.roll(compute_cmp_state_fn)

    for param, exact_solution in zip(params, Toy2dCMP_problem_properties["exact_solution"]):
        assert torch.allclose(param, exact_solution)
