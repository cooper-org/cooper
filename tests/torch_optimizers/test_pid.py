import cooper_test_utils
import pytest
import torch

import cooper
from cooper.multipliers import IndexedMultiplier
from cooper.optim import PID

# TODO(juan43ramirez): test with multiple parameter groups

ALL_HYPER_PARAMS = [
    (0, 1, 0, 0),  # I controller
    (1, 1, 0, 0),  # PI controller
    (0, 1, 1, 0),  # PD controller
    (1, 1, 1, 0),  # PID controller
    (1, 1, 1, 0.9),  # PID controller with EMA
]


@pytest.mark.parametrize(["Kp", "Ki", "Kd", "ema_nu"], ALL_HYPER_PARAMS)
def test_manual_pid(Kp, Ki, Kd, ema_nu):
    param = torch.tensor([1.0], requires_grad=True)

    loss_fn = lambda param: param**2 / 2

    lr = 0.1

    def compute_gradient():
        # For the quadratic loss, the gradient is simply the current value of p.
        return param.clone().detach()

    def update_ema(ema, value):
        return ema_nu * ema + (1 - ema_nu) * value

    def recursive_PID_direction(error, error_change, delta_change):
        return Kp * error_change + Ki * error + Kd * delta_change

    optimizer = PID([param], lr=lr, Kp=Kp, Ki=Ki, Kd=Kd, ema_nu=ema_nu, maximize=False)

    def do_optimizer_step():
        optimizer.zero_grad()
        loss = loss_fn(param)
        loss.backward()
        optimizer.step()

    # Initialization of PID hyperparameters
    error_minus_1 = torch.zeros_like(param)
    delta_minus_1 = torch.zeros_like(param)
    p0 = param.clone().detach()

    # -------------------------------------------- First step of optimization
    # Manual first step of optimization. The gradient is simply the current value of p.
    error_0 = -compute_gradient()
    error_change_0 = error_0 - error_minus_1
    delta_0 = update_ema(delta_minus_1, error_change_0)
    delta_change_0 = delta_0 - delta_minus_1
    p1 = p0 + lr * recursive_PID_direction(error_0, error_change_0, delta_change_0)

    do_optimizer_step()

    # Check that iterates match
    assert torch.allclose(param, p1)

    # When entering the next iteration, the "current" error and delta become the "previous"
    if Kp != 0 or Ki != 0:
        assert torch.allclose(optimizer.state[param]["previous_error"], error_0)
    if Kd != 0:
        assert torch.allclose(optimizer.state[param]["previous_delta"], delta_0)

    # -------------------------------------------- Second step of optimization
    error_1 = -compute_gradient()
    error_change_1 = error_1 - error_0
    delta_1 = update_ema(delta_0, error_change_1)
    delta_change_1 = delta_1 - delta_0
    p2 = p1 + lr * recursive_PID_direction(error_1, error_change_1, delta_change_1)

    do_optimizer_step()

    # Check that iterates match
    assert torch.allclose(param, p2)

    # When entering the next iteration, the "current" error and delta become the "previous"
    if Kp != 0 or Ki != 0:
        assert torch.allclose(optimizer.state[param]["previous_error"], error_1)
    if Kd != 0:
        assert torch.allclose(optimizer.state[param]["previous_delta"], delta_1)

    # -------------------------------------------- Third step of optimization
    error_2 = -compute_gradient()
    error_change_2 = error_2 - error_1
    delta_2 = update_ema(delta_1, error_change_2)
    delta_change_2 = delta_2 - delta_1
    p3 = p2 + lr * recursive_PID_direction(error_2, error_change_2, delta_change_2)

    do_optimizer_step()

    # Check that iterates match
    assert torch.allclose(param, p3)

    # When entering the next iteration, the "current" error and delta become the "previous"
    if Kp != 0 or Ki != 0:
        assert torch.allclose(optimizer.state[param]["previous_error"], error_2)
    if Kd != 0:
        assert torch.allclose(optimizer.state[param]["previous_delta"], delta_2)


@pytest.mark.parametrize(["Kp", "Ki", "Kd", "ema_nu"], ALL_HYPER_PARAMS)
def test_manual_sparse_pid(Kp, Ki, Kd, ema_nu):
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

    optimizer = PID(param.parameters(), lr=lr, Kp=Kp, Ki=Ki, Kd=Kd, ema_nu=ema_nu)

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
    previous_error = torch.zeros_like(p0)
    previous_error[first_indices] = p0.clone()[first_indices]
    # For the first step, the `previous_delta` is 0 for all indices
    previous_delta = torch.zeros_like(p0)

    do_step(first_indices)

    assert torch.allclose(param(all_indices), p1)

    # Check the state of the optimizer. For observer indices, should contain the first
    # gradient in `previous_error` and also in `previous_delta`. 0 otherwise.
    state = optimizer.state[list(param.parameters())[0]]

    assert torch.allclose(state["previous_error"].flatten(), previous_error)
    assert torch.allclose(state["previous_delta"].flatten(), previous_delta)

    # -------------------------------------------- Second step of optimization
    second_indices = torch.tensor([0, 3, 5, 7, 9], device=device)

    # Manual second step
    p2 = p1.clone()
    change = p1 - previous_error

    unseen_indices = torch.ones_like(all_indices, dtype=torch.bool)
    unseen_indices[first_indices] = False
    change[unseen_indices] = 0.0

    p2[second_indices] -= lr * PID_direction(p1, change, 0.0)[second_indices]

    previous_error[second_indices] = p1.clone()[second_indices]

    # For indices that were observed before *and* now, the previous_delta is p1 - p0.
    # Note that no EMA calculation is being applied yet.
    twice_observed_indices = [idx for idx in range(10) if idx in first_indices and idx in second_indices]
    previous_delta[twice_observed_indices] = p1[twice_observed_indices] - p0[twice_observed_indices]

    do_step(second_indices)

    assert torch.allclose(param(all_indices), p2)

    state = optimizer.state[list(param.parameters())[0]]

    assert torch.allclose(state["previous_error"].flatten(), previous_error)
    assert torch.allclose(state["previous_delta"].flatten(), previous_delta)

    #  -------------------------------------------- Third step of optimization
    third_indices = torch.tensor([0, 2, 4, 7], device=device)

    # Manual third step
    p3 = p2.clone()

    # Change is simply p2 - previous_error
    change = p2 - previous_error

    # Calculate the previous_delta
    new_previous_delta = previous_delta.clone()
    # * For indices observed only once in the past, and also now:
    #  * If they are part of first_indices, the previous_delta is p2 - p0
    observed_now_and_first = [idx for idx in range(10) if idx in first_indices and idx in third_indices]
    new_previous_delta[observed_now_and_first] = p2[observed_now_and_first] - p0[observed_now_and_first]
    #  * If they are part of second_indices, the previous_delta is p2 - p1
    observed_now_and_second = [idx for idx in range(10) if idx in second_indices and idx in third_indices]
    new_previous_delta[observed_now_and_second] = p2[observed_now_and_second] - p1[observed_now_and_second]
    # * For indices observed twice in the past, previous_delta is:
    #   ema_nu * previous_delta + (1 - ema_nu) * change
    observed_thrice = [
        idx for idx in range(10) if idx in first_indices and idx in second_indices and idx in third_indices
    ]

    new_previous_delta[observed_thrice] = (
        ema_nu * previous_delta[observed_thrice] + (1 - ema_nu) * change[observed_thrice]
    )

    # Although the new_previous_delta is calculated for indices from `observer_now_and_first`
    # and `observer_now_and_second`, since the old previous_delta was 0 for those indices,
    # curvature is not estimated for them.
    curvature = torch.zeros_like(p2)
    curvature[observed_thrice] = new_previous_delta[observed_thrice] - previous_delta[observed_thrice]

    p3[third_indices] -= lr * PID_direction(p2, change, curvature)[third_indices]

    do_step(third_indices)
    state = optimizer.state[list(param.parameters())[0]]

    assert torch.allclose(param(all_indices), p3)

    previous_error[third_indices] = p2.clone()[third_indices]

    assert torch.allclose(state["previous_error"].flatten(), previous_error)
    assert torch.allclose(state["previous_delta"].flatten(), new_previous_delta)


@pytest.mark.parametrize(["Kp", "Ki", "Kd", "ema_nu"], ALL_HYPER_PARAMS)
def test_pid_convergence(
    Kp, Ki, Kd, ema_nu, Toy2dCMP_problem_properties, Toy2dCMP_params_init, use_multiple_primal_optimizers, device
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
    dual_optimizer = PID(dual_params, lr=0.01, Kp=Kp, Ki=Ki, Kd=Kd, ema_nu=ema_nu, maximize=True)

    cooper_optimizer = cooper.optim.SimultaneousOptimizer(
        primal_optimizers, dual_optimizer, constraint_groups=cmp.constraint_groups
    )

    compute_cmp_state_fn = lambda: cmp.compute_cmp_state(params)

    for step_id in range(1500):
        cooper_optimizer.roll(compute_cmp_state_fn)

    for param, exact_solution in zip(params, Toy2dCMP_problem_properties["exact_solution"]):
        assert torch.allclose(param, exact_solution)
