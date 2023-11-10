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
    error_0 = -compute_gradient()
    error_change_0 = error_0 - error_minus_1
    if Kd != 0:
        delta_0 = update_ema(delta_minus_1, error_change_0)
        delta_change_0 = delta_0 - delta_minus_1
    else:
        delta_change_0 = 0.0
    p1 = p0 + lr * recursive_PID_direction(error_0, error_change_0, delta_change_0)

    do_optimizer_step()

    # Check that iterates match
    assert torch.allclose(param, p1)

    # When entering the next iteration, the "current" error and delta become the "previous"
    if Kp != 0 or Kd != 0:
        assert torch.allclose(optimizer.state[param]["previous_error"], error_0)
    if Kd != 0:
        assert torch.allclose(optimizer.state[param]["previous_delta"], delta_0)

    # -------------------------------------------- Second step of optimization
    error_1 = -compute_gradient()
    error_change_1 = error_1 - error_0
    if Kd != 0:
        delta_1 = update_ema(delta_0, error_change_1)
        delta_change_1 = delta_1 - delta_0
    else:
        delta_change_1 = 0.0
    p2 = p1 + lr * recursive_PID_direction(error_1, error_change_1, delta_change_1)

    do_optimizer_step()

    # Check that iterates match
    assert torch.allclose(param, p2)

    # When entering the next iteration, the "current" error and delta become the "previous"
    if Kp != 0 or Kd != 0:
        assert torch.allclose(optimizer.state[param]["previous_error"], error_1)
    if Kd != 0:
        assert torch.allclose(optimizer.state[param]["previous_delta"], delta_1)

    # -------------------------------------------- Third step of optimization
    error_2 = -compute_gradient()
    error_change_2 = error_2 - error_1
    if Kd != 0:
        delta_2 = update_ema(delta_1, error_change_2)
        delta_change_2 = delta_2 - delta_1
    else:
        delta_change_2 = 0.0
    p3 = p2 + lr * recursive_PID_direction(error_2, error_change_2, delta_change_2)

    do_optimizer_step()

    # Check that iterates match
    assert torch.allclose(param, p3)

    # When entering the next iteration, the "current" error and delta become the "previous"
    if Kp != 0 or Kd != 0:
        assert torch.allclose(optimizer.state[param]["previous_error"], error_2)
    if Kd != 0:
        assert torch.allclose(optimizer.state[param]["previous_delta"], delta_2)


@pytest.mark.parametrize(["Kp", "Ki", "Kd", "ema_nu"], ALL_HYPER_PARAMS)
def test_manual_sparse_pid(Kp, Ki, Kd, ema_nu):
    if not torch.cuda.is_available():
        pytest.skip("Sparse gradients are only supported on GPU.")
    device = "cuda"

    num_multipliers = 10
    multiplier_init = torch.ones(num_multipliers, 1, device=device)
    multiplier_module = IndexedMultiplier(init=multiplier_init)
    param = multiplier_module.weight

    def loss_fn(indices):
        return multiplier_module(indices).pow(2).sum() / 2

    lr = 0.1

    def compute_gradient(indices):
        # For the quadratic loss, the gradient is simply the current value of p.
        return multiplier_module(indices).reshape(-1, 1).clone().detach()

    def update_ema(ema, value):
        return ema_nu * ema + (1 - ema_nu) * value

    def recursive_PID_direction(error, error_change, delta_change):
        return Kp * error_change + Ki * error + Kd * delta_change

    optimizer = PID(multiplier_module.parameters(), lr=lr, Kp=Kp, Ki=Ki, Kd=Kd, ema_nu=ema_nu, maximize=False)

    def do_optimizer_step(indices):
        optimizer.zero_grad()
        loss = loss_fn(indices)
        loss.backward()
        optimizer.step()

    # Initialization of PID hyperparameters
    error_minus_1 = torch.zeros_like(param)
    delta_minus_1 = torch.zeros_like(param)
    p0 = param.clone().detach()

    # -------------------------------------------- First step of optimization
    selected_indices = torch.tensor([0, 1, 2, 4, 7], device=device)
    not_selected_indices_mask = torch.ones(param.shape[0], dtype=bool)
    not_selected_indices_mask[selected_indices] = False

    previous_error = error_minus_1
    error_0 = -compute_gradient(selected_indices)
    error_change_0 = error_0 - previous_error[selected_indices]
    if Kd != 0:
        previous_delta = delta_minus_1
        delta_0 = update_ema(previous_delta[selected_indices], error_change_0)
        delta_change_0 = delta_0 - previous_delta[selected_indices]
    else:
        delta_change_0 = 0.0
    p1 = p0.clone()
    p1[selected_indices] += lr * recursive_PID_direction(error_0, error_change_0, delta_change_0)

    do_optimizer_step(selected_indices)

    assert torch.allclose(param, p1)

    # When entering the next iteration, the "current" error and delta become the "previous"
    # Check that entries in both modified and unmodified indices correspond to the correct values
    if Kp != 0 or Kd != 0:
        print(optimizer.state[param].keys())
        buffer = optimizer.state[param]["previous_error"]
        assert torch.allclose(buffer[selected_indices], error_0)
        assert torch.allclose(buffer[not_selected_indices_mask], error_minus_1[not_selected_indices_mask])
    if Kd != 0:
        buffer = optimizer.state[param]["previous_delta"]
        assert torch.allclose(buffer[selected_indices], delta_0)
        assert torch.allclose(buffer[not_selected_indices_mask], delta_minus_1[not_selected_indices_mask])

    # -------------------------------------------- First step of optimization
    selected_indices = torch.tensor([3, 5, 7, 8, 9], device=device)
    not_selected_indices_mask = torch.ones(param.shape[0], dtype=bool)
    not_selected_indices_mask[selected_indices] = False

    if Kp != 0 or Kd != 0:
        previous_error = optimizer.state[param]["previous_error"].clone()
    else:
        previous_error = torch.zeros_like(param)
    error_1 = -compute_gradient(selected_indices)
    error_change_1 = error_1 - previous_error[selected_indices]
    if Kd != 0:
        previous_delta = optimizer.state[param]["previous_delta"]
        delta_1 = update_ema(previous_delta[selected_indices], error_change_1)
        delta_change_1 = delta_1 - previous_delta[selected_indices]
    else:
        delta_change_1 = 0.0
    p2 = p1.clone()
    p2[selected_indices] += lr * recursive_PID_direction(error_1, error_change_1, delta_change_1)

    do_optimizer_step(selected_indices)

    assert torch.allclose(param, p2)

    # When entering the next iteration, the "current" error and delta become the "previous"
    # Check that entries in both modified and unmodified indices correspond to the correct values
    if Kp != 0 or Kd != 0:
        print(optimizer.state[param].keys())
        buffer = optimizer.state[param]["previous_error"]
        assert torch.allclose(buffer[selected_indices], error_1)
        assert torch.allclose(buffer[not_selected_indices_mask], previous_error[not_selected_indices_mask])
    if Kd != 0:
        buffer = optimizer.state[param]["previous_delta"]
        assert torch.allclose(buffer[selected_indices], delta_1)
        assert torch.allclose(buffer[not_selected_indices_mask], previous_delta[not_selected_indices_mask])


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

    dual_params = [{"params": _.parameters()} for _ in cmp.multipliers]
    dual_optimizer = PID(dual_params, lr=0.01, Kp=Kp, Ki=Ki, Kd=Kd, ema_nu=ema_nu, maximize=True)

    cooper_optimizer = cooper.optim.SimultaneousOptimizer(
        primal_optimizers, dual_optimizer, multipliers=cmp.multipliers
    )

    compute_cmp_state_fn = lambda: cmp.compute_cmp_state(params)

    for step_id in range(1500):
        cooper_optimizer.roll(compute_cmp_state_fn)

    for param, exact_solution in zip(params, Toy2dCMP_problem_properties["exact_solution"]):
        assert torch.allclose(param, exact_solution)
