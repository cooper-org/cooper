import cooper_test_utils
import pytest
import torch

import cooper
from cooper.multipliers import IndexedMultiplier
from cooper.optim import PID

# TODO(juan43ramirez): test with multiple parameter groups

# Configurations correspond to: [I, PI, PD, PID, PID-EMA]
ALL_HYPER_PARAMS = [(0, 1, 0, 0), (1, 1, 0, 0), (0, 1, 1, 0), (1, 1, 1, 0), (1, 1, 1, 0.9)]


@pytest.mark.parametrize(["Kp", "Ki", "Kd", "ema_beta"], ALL_HYPER_PARAMS)
def test_manual_pid(Kp, Ki, Kd, ema_beta):
    param = torch.tensor([1.0], requires_grad=True)

    loss_fn = lambda param: param**2 / 2

    lr = 0.1

    def update_ema(ema, value):
        return ema_beta * ema + (1 - ema_beta) * value

    def PID_direction(grad, change, curvature):
        return Ki * grad + Kp * change + Kd * curvature

    optimizer = PID([param], lr=lr, Kp=Kp, Ki=Ki, Kd=Kd, ema_beta=ema_beta)

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
    # `previous_grad`, but not `delta_ema` yet.
    assert torch.allclose(optimizer.state[param]["previous_grad"], p0)
    assert "delta_ema" not in optimizer.state[param]

    # -------------------------------------------- Second step of optimization
    # The delta_ema is initialized to the difference of the first two gradients.
    # However, it is not used for the second step of optimization.
    delta_ema = p1 - p0
    p2 = p1 - lr * PID_direction(p1, p1 - p0, 0.0)

    do_step()

    assert torch.allclose(param, p2)

    # The state contain p1 in `previous_grad` and p1 - p0 in `delta_ema`.
    assert torch.allclose(optimizer.state[param]["previous_grad"], p1)
    assert torch.allclose(optimizer.state[param]["delta_ema"], delta_ema)

    # -------------------------------------------- Third step of optimization
    new_delta_ema = update_ema(delta_ema, p2 - p1)
    p3 = p2 - lr * PID_direction(p2, p2 - p1, new_delta_ema - delta_ema)

    do_step()

    assert torch.allclose(param, p3)

    # The state contain p2 in `previous_grad` and p2 - p1 in `delta_ema`.
    assert torch.allclose(optimizer.state[param]["previous_grad"], p2)
    assert torch.allclose(optimizer.state[param]["delta_ema"], new_delta_ema)


@pytest.mark.parametrize(["Kp", "Ki", "Kd", "ema_beta"], ALL_HYPER_PARAMS)
def test_manual_sparse_pid(Kp, Ki, Kd, ema_beta):
    if not torch.cuda.is_available():
        pytest.skip("Sparse gradients are only supported on GPU.")

    device = "cuda"

    param = IndexedMultiplier(torch.ones(10, 1, device=device), use_sparse_gradient=True)
    all_indices = torch.arange(10, device=device)

    def loss_fn(param, index):
        return param(index).pow(2).sum() / 2

    lr = 0.1

    def PID_direction(grad, change, curvature):
        return Ki * grad + Kp * change + Kd * curvature

    optimizer = PID(param.parameters(), lr=lr, Kp=Kp, Ki=Ki, Kd=Kd, ema_beta=ema_beta)

    def do_step(indices):
        optimizer.zero_grad()
        loss = loss_fn(param, indices)
        loss.backward()
        optimizer.step()

    # -------------------------------------------- First step of optimization
    first_indices = torch.tensor([0, 1, 2, 4, 7], device=device)

    # Manual first step of optimization. The gradient is simply the current value of p.
    p0 = param(all_indices).clone().detach()
    p1 = p0.clone()
    p1[first_indices] -= lr * PID_direction(p0, 0.0, 0.0)[first_indices]  # only update the current indices

    # Manually calculate the state of the optimizer
    previous_grad = torch.zeros_like(p0)
    previous_grad[first_indices] = p0.clone()[first_indices]
    # For the first step, the `delta_ema` is 0 for all indices
    delta_ema = torch.zeros_like(p0)

    do_step(first_indices)

    assert torch.allclose(param(all_indices), p1)

    # Check the state of the optimizer. For observer indices, should contain the first
    # gradient in `previous_grad` and also in `delta_ema`. 0 otherwise.
    state = optimizer.state[list(param.parameters())[0]]

    assert torch.allclose(state["previous_grad"].flatten(), previous_grad)
    assert torch.allclose(state["delta_ema"].flatten(), delta_ema)

    # -------------------------------------------- Second step of optimization
    second_indices = torch.tensor([0, 3, 5, 7, 9], device=device)

    # Manual second step
    p2 = p1.clone()
    change = p1 - previous_grad

    unseen_indices = torch.ones_like(all_indices, dtype=torch.bool)
    unseen_indices[first_indices] = False
    change[unseen_indices] = 0.0

    p2[second_indices] -= lr * PID_direction(p1, change, 0.0)[second_indices]

    previous_grad[second_indices] = p1.clone()[second_indices]

    # For indices that were observed before *and* now, the delta_ema is p1 - p0.
    # Note that no EMA calculation is being applied yet.
    twice_observed_indices = [idx for idx in range(10) if idx in first_indices and idx in second_indices]
    delta_ema[twice_observed_indices] = p1[twice_observed_indices] - p0[twice_observed_indices]

    do_step(second_indices)

    assert torch.allclose(param(all_indices), p2)

    state = optimizer.state[list(param.parameters())[0]]

    assert torch.allclose(state["previous_grad"].flatten(), previous_grad)
    assert torch.allclose(state["delta_ema"].flatten(), delta_ema)

    #  -------------------------------------------- Third step of optimization
    third_indices = torch.tensor([0, 2, 4, 7], device=device)

    # Manual third step
    p3 = p2.clone()

    # Change is simply p2 - previous_grad
    change = p2 - previous_grad

    # Calculate the delta_ema
    new_delta_ema = delta_ema.clone()
    # * For indices observed only once in the past, and also now:
    #  * If they are part of first_indices, the delta_ema is p2 - p0
    observed_now_and_first = [idx for idx in range(10) if idx in first_indices and idx in third_indices]
    new_delta_ema[observed_now_and_first] = p2[observed_now_and_first] - p0[observed_now_and_first]
    #  * If they are part of second_indices, the delta_ema is p2 - p1
    observed_now_and_second = [idx for idx in range(10) if idx in second_indices and idx in third_indices]
    new_delta_ema[observed_now_and_second] = p2[observed_now_and_second] - p1[observed_now_and_second]
    # * For indices observed twice in the past, delta_ema is:
    #   ema_beta * delta_ema + (1 - ema_beta) * change
    observed_thrice = [
        idx for idx in range(10) if idx in first_indices and idx in second_indices and idx in third_indices
    ]

    new_delta_ema[observed_thrice] = ema_beta * delta_ema[observed_thrice] + (1 - ema_beta) * change[observed_thrice]

    # Although the new_delta_ema is calculated for indices from `observer_now_and_first`
    # and `observer_now_and_second`, since the old delta_ema was 0 for those indices,
    # curvature is not estimated for them.
    curvature = torch.zeros_like(p2)
    curvature[observed_thrice] = new_delta_ema[observed_thrice] - delta_ema[observed_thrice]

    p3[third_indices] -= lr * PID_direction(p2, change, curvature)[third_indices]

    do_step(third_indices)
    state = optimizer.state[list(param.parameters())[0]]

    assert torch.allclose(param(all_indices), p3)

    previous_grad[third_indices] = p2.clone()[third_indices]

    assert torch.allclose(state["previous_grad"].flatten(), previous_grad)
    assert torch.allclose(state["delta_ema"].flatten(), new_delta_ema)


@pytest.mark.parametrize(["Kp", "Ki", "Kd", "ema_beta"], ALL_HYPER_PARAMS)
def test_pid_convergence(
    Kp, Ki, Kd, ema_beta, Toy2dCMP_problem_properties, Toy2dCMP_params_init, use_multiple_primal_optimizers, device
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
    dual_optimizer = PID(dual_params, lr=0.01, Kp=Kp, Ki=Ki, Kd=Kd, ema_beta=ema_beta, maximize=True)

    cooper_optimizer = cooper.optim.SimultaneousOptimizer(
        primal_optimizers, dual_optimizer, constraint_groups=cmp.constraint_groups
    )

    compute_cmp_state_fn = lambda: cmp.compute_cmp_state(params)

    for step_id in range(1500):
        cooper_optimizer.roll(compute_cmp_state_fn)

    for param, exact_solution in zip(params, Toy2dCMP_problem_properties["exact_solution"]):
        assert torch.allclose(param, exact_solution)
