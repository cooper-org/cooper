import cooper_test_utils
import pytest
import torch

import cooper
from cooper.multipliers import IndexedMultiplier
from cooper.optim import PID
from cooper.optim.PID_optimizer import PIDInitType

# TODO(juan43ramirez): test with multiple parameter groups

# Parameter order is: Kp, Ki, Kd, ema_nu
ALL_HYPER_PARAMS = [
    (0, 1, 0, 0),  # I controller
    (1, 1, 0, 0),  # PI controller
    (0, 1, 1, 0),  # PD controller
    (1, 1, 1, 0),  # PID controller
    (1, 1, 1, 0.9),  # PID controller with EMA
]

LR = 0.01


@pytest.mark.parametrize(["Kp", "Ki", "Kd", "ema_nu"], ALL_HYPER_PARAMS)
@pytest.mark.parametrize("maximize", [True, False])
def test_dense_pid_zero_init(Kp, Ki, Kd, ema_nu, maximize):
    param = torch.tensor([1.0], requires_grad=True)

    loss_fn = lambda param: param**2 / 2

    def compute_gradient():
        # For the quadratic loss, the gradient is simply the current value of p.
        return param.clone().detach()

    def recursive_PID_direction(error, error_change, delta_change):
        return Kp * error_change + Ki * error + Kd * delta_change

    update_sign = 1 if maximize else -1
    optimizer = PID([param], lr=LR, Kp=Kp, Ki=Ki, Kd=Kd, ema_nu=ema_nu, maximize=maximize, init_type=PIDInitType.ZEROS)

    def do_optimizer_step():
        optimizer.zero_grad()
        loss = loss_fn(param)
        loss.backward()
        optimizer.step()

    # Initialization of PID hyperparameters
    error_minus_1 = torch.zeros_like(param)
    previous_error = error_minus_1
    delta_minus_1 = torch.zeros_like(param)
    previous_delta = delta_minus_1
    previous_param = param.clone().detach()

    for step_id in range(1, 10):
        new_error = compute_gradient()
        error_change = new_error - previous_error
        new_delta = ema_nu * previous_delta + (1 - ema_nu) * error_change
        delta_change = new_delta - previous_delta
        param_update = recursive_PID_direction(new_error, error_change, delta_change).clone().detach()
        new_param = previous_param + update_sign * LR * param_update

        do_optimizer_step()

        # Check that iterates match
        assert torch.allclose(param, new_param)

        # When entering the next iteration, the "current" error and delta become the "previous"
        if Kp != 0 or Kd != 0:
            assert torch.allclose(optimizer.state[param]["previous_error"], new_error)
        if Kd != 0:
            assert torch.allclose(optimizer.state[param]["previous_delta"], new_delta)

        previous_error = new_error
        previous_delta = new_delta
        previous_param = new_param


@pytest.mark.parametrize(["Kp", "Ki", "Kd", "ema_nu"], ALL_HYPER_PARAMS)
@pytest.mark.parametrize("maximize", [True, False])
def test_dense_pid_gd_init_matches_pi_first_two_steps(Kp, Ki, Kd, ema_nu, maximize):
    """This test checks that the first *two* steps of `PID(KP, KI, KD, init_type=SGD_PI)`
    are equivalent to PI(KP, KI)."""

    param_pi = torch.tensor([1.0], requires_grad=True)
    param_pid = torch.tensor([1.0], requires_grad=True)

    loss_fn = lambda param: param**2 / 2

    def do_optimizer_step(_param, _optimizer):
        _optimizer.zero_grad()
        loss = loss_fn(_param)
        loss.backward()
        _optimizer.step()

    pid_optimizer = PID(
        [param_pid], lr=LR, Kp=Kp, Ki=Ki, Kd=Kd, ema_nu=ema_nu, maximize=maximize, init_type=PIDInitType.SGD_PI
    )
    pi_optimizer = PID([param_pi], lr=LR, Kp=Kp, Ki=Ki, maximize=maximize)

    for step_id in range(1, 2 + 1):
        do_optimizer_step(param_pi, pi_optimizer)
        do_optimizer_step(param_pid, pid_optimizer)
        assert torch.allclose(param_pi, param_pid)


@pytest.mark.parametrize(["Kp", "Ki", "Kd", "ema_nu"], ALL_HYPER_PARAMS)
@pytest.mark.parametrize("maximize", [True, False])
def test_dense_pid_no_kd_matches_pi(Kp, Ki, Kd, ema_nu, maximize):
    """This test checks that the first 100 steps of `PID(KP, KI, KD=0, init_type=SGD_PI)`
    are equivalent to PI(KP, KI)."""

    if Kd != 0:
        pytest.skip("Test only applies to PID with KD gain equal zero")

    param_pi = torch.tensor([1.0], requires_grad=True)
    param_pid = torch.tensor([1.0], requires_grad=True)

    loss_fn = lambda param: param**2 / 2

    def do_optimizer_step(_param, _optimizer):
        _optimizer.zero_grad()
        loss = loss_fn(_param)
        loss.backward()
        _optimizer.step()

    pid_optimizer = PID(
        [param_pid], lr=LR, Kp=Kp, Ki=Ki, Kd=Kd, ema_nu=ema_nu, maximize=maximize, init_type=PIDInitType.SGD_PI
    )
    pi_optimizer = PID([param_pi], lr=LR, Kp=Kp, Ki=Ki, maximize=maximize)

    for step_id in range(100):
        do_optimizer_step(param_pi, pi_optimizer)
        do_optimizer_step(param_pid, pid_optimizer)
        assert torch.allclose(param_pi, param_pid)


@pytest.mark.parametrize(["Kp", "Ki", "Kd", "ema_nu"], ALL_HYPER_PARAMS)
@pytest.mark.parametrize("maximize", [True, False])
def test_sparse_pid_update_zeros_init(Kp, Ki, Kd, ema_nu, maximize, device):
    num_multipliers = 10
    multiplier_init = torch.ones(num_multipliers, 1, device=device)
    multiplier_module = IndexedMultiplier(init=multiplier_init, constraint_type=cooper.ConstraintType.EQUALITY)
    param = multiplier_module.weight

    def loss_fn(indices):
        return multiplier_module(indices).pow(2).sum() / 2

    update_sign = 1 if maximize else -1

    def compute_analytic_gradient(indices):
        # For the quadratic loss, the gradient is simply the current value of p.
        return multiplier_module(indices).reshape(-1, 1).clone().detach()

    def recursive_PID_direction(error, error_change, delta_change):
        return Kp * error_change + Ki * error + Kd * delta_change

    optimizer = PID(
        multiplier_module.parameters(),
        lr=LR,
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        ema_nu=ema_nu,
        maximize=maximize,
        init_type=PIDInitType.ZEROS,
    )

    def do_optimizer_step(indices):
        optimizer.zero_grad()
        loss = loss_fn(indices)
        loss.backward()
        optimizer.step()

    # Initialization of PID hyperparameters
    prev_error_buffer = torch.zeros_like(param)
    prev_delta_buffer = torch.zeros_like(param)

    # ------------------------ First step ------------------------
    selected_indices_0 = torch.tensor([0, 1, 2, 3], device=device)
    selected_indices_mask_0 = torch.zeros_like(param, dtype=bool)
    selected_indices_mask_0[selected_indices_0] = True

    error_0 = compute_analytic_gradient(selected_indices_0)
    error_change_0 = error_0 - prev_error_buffer[selected_indices_0]
    delta_0 = ema_nu * prev_delta_buffer[selected_indices_0] + (1 - ema_nu) * error_change_0
    delta_change_0 = delta_0 - prev_delta_buffer[selected_indices_0]
    prev_error_buffer[selected_indices_0] = error_0.clone().detach()
    prev_delta_buffer[selected_indices_0] = delta_0.clone().detach()

    new_param = param.clone().detach()
    pid_update = update_sign * LR * recursive_PID_direction(error_0, error_change_0, delta_change_0)
    new_param[selected_indices_0] += pid_update.clone().detach()

    do_optimizer_step(selected_indices_0)

    assert torch.allclose(param, new_param)

    if Kp != 0:
        buffer = optimizer.state[param]["previous_error"]
        # Check that entries in modified and unmodified indices have the right values
        assert torch.allclose(buffer[selected_indices_mask_0], error_0)
        assert torch.allclose(buffer[~selected_indices_mask_0], torch.zeros_like(buffer[~selected_indices_mask_0]))

    # # ------------------------ Second step ------------------------
    # Note that index 3 was already seen in the previous step, so we don't have to
    # initialize its buffer entry. Indices 4, 8, 9 are unseen.
    selected_indices_1 = torch.tensor([3, 5, 6, 7], device=device)
    selected_indices_mask_1 = torch.zeros_like(param, dtype=bool)
    selected_indices_mask_1[selected_indices_1] = True

    error_1 = compute_analytic_gradient(selected_indices_1)
    error_change_1 = error_1 - prev_error_buffer[selected_indices_1]
    delta_1 = ema_nu * prev_delta_buffer[selected_indices_1] + (1 - ema_nu) * error_change_1
    delta_change_1 = delta_1 - prev_delta_buffer[selected_indices_1]
    prev_error_buffer[selected_indices_1] = error_1.clone().detach()
    prev_delta_buffer[selected_indices_1] = delta_1.clone().detach()

    new_param = param.clone().detach()
    pid_update = update_sign * LR * recursive_PID_direction(error_1, error_change_1, delta_change_1)
    new_param[selected_indices_1] += pid_update.clone().detach()

    do_optimizer_step(selected_indices_1)

    assert torch.allclose(param, new_param)

    if Kp != 0:
        buffer = optimizer.state[param]["previous_error"]
        # Check that entries in modified and unmodified indices have the right values
        assert torch.allclose(buffer[selected_indices_mask_0][:-1], error_0)
        assert torch.allclose(buffer[selected_indices_mask_1].reshape(error_1.shape), error_1)
        unseen_indices = torch.tensor([4, 8, 9], device=device)
        assert torch.allclose(buffer[unseen_indices], torch.zeros_like(buffer[unseen_indices]))


@pytest.mark.parametrize(["Kp", "Ki", "Kd", "ema_nu"], ALL_HYPER_PARAMS)
@pytest.mark.parametrize("maximize", [True, False])
def test_manual_sparse_pid_update_sgd_pi_init(Kp, Ki, Kd, ema_nu, maximize, device):
    num_multipliers = 10
    multiplier_init = torch.ones(num_multipliers, 1, device=device)
    multiplier_module = IndexedMultiplier(init=multiplier_init, constraint_type=cooper.ConstraintType.EQUALITY)
    param = multiplier_module.weight

    def loss_fn(indices):
        return multiplier_module(indices).pow(2).sum() / 2

    update_sign = 1 if maximize else -1

    def compute_analytic_gradient(indices):
        # For the quadratic loss, the gradient is simply the current value of p.
        return multiplier_module(indices).reshape(-1, 1).clone().detach()

    optimizer = PID(
        multiplier_module.parameters(),
        lr=LR,
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        ema_nu=ema_nu,
        maximize=maximize,
        init_type=PIDInitType.SGD_PI,
    )

    def do_optimizer_step(indices):
        optimizer.zero_grad()
        loss = loss_fn(indices)
        loss.backward()
        optimizer.step()

    # ------------------------ First step ------------------------
    selected_indices_0 = torch.tensor([0, 1, 2, 3], device=device)
    selected_indices_mask_0 = torch.zeros_like(param, dtype=bool)
    selected_indices_mask_0[selected_indices_0] = True

    error_0 = compute_analytic_gradient(selected_indices_0)

    # First step is designed to match a plain SGD update
    new_param = param.clone().detach()
    pid_update = update_sign * LR * error_0
    new_param[selected_indices_0] += pid_update.clone().detach()

    do_optimizer_step(selected_indices_0)

    assert torch.allclose(param, new_param)

    if Kp != 0:
        assert not torch.any(optimizer.state[param]["needs_error_initialization_mask"][selected_indices_0])
        buffer = optimizer.state[param]["previous_error"]
        # Check that entries in modified and unmodified indices have the right values
        assert torch.allclose(buffer[selected_indices_mask_0], error_0)
        assert torch.allclose(buffer[~selected_indices_mask_0], torch.zeros_like(buffer[~selected_indices_mask_0]))

    if Kd != 0:
        # No entries should have a delta buffer initialized at this point -- all need to
        # be initialized still
        assert torch.all(optimizer.state[param]["needs_delta_initialization_mask"])

    # # ------------------------ Second step ------------------------
    # Note that index 3 was already seen in the previous step, so we don't have to
    # initialize its buffer entry. Indices 4, 8, 9 are unseen.
    selected_indices_1 = torch.tensor([3, 5, 6, 7], device=device)
    selected_indices_mask_1 = torch.zeros_like(param, dtype=bool)
    selected_indices_mask_1[selected_indices_1] = True

    error_1 = compute_analytic_gradient(selected_indices_1)

    new_param = param.clone().detach()
    # Since index 3 was already seen, the PID update is constructed to be a PI step
    error_change_param3 = error_1.clone().detach()[0] - error_0.clone().detach()[-1]
    new_param[3] += update_sign * LR * (Ki * error_1.clone().detach()[0] + Kp * error_change_param3)
    # All other indices are updated with an SGD-like step
    new_param[[5, 6, 7]] += update_sign * LR * Ki * error_1.clone().detach()[1:]

    do_optimizer_step(selected_indices_1)

    assert torch.allclose(param, new_param)

    if Kp != 0:
        assert not torch.any(optimizer.state[param]["needs_error_initialization_mask"][selected_indices_1])
        buffer = optimizer.state[param]["previous_error"]
        # Check that entries in modified and unmodified indices have the right values
        assert torch.allclose(buffer[selected_indices_mask_0][:-1], error_0)
        assert torch.allclose(buffer[selected_indices_mask_1].reshape(error_1.shape), error_1)
        unseen_indices = torch.tensor([4, 8, 9], device=device)
        assert torch.allclose(buffer[unseen_indices], torch.zeros_like(buffer[unseen_indices]))

    if Kd != 0:
        # Only index 3 should have a delta buffer initialized at this point
        assert not optimizer.state[param]["needs_delta_initialization_mask"][3]
        assert torch.all(optimizer.state[param]["needs_delta_initialization_mask"][[0, 1, 2, 4, 5, 6, 7, 8, 9]])


@pytest.mark.parametrize(["Kp", "Ki", "Kd", "ema_nu"], ALL_HYPER_PARAMS)
@pytest.mark.parametrize("pid_init_type", [PIDInitType.ZEROS, PIDInitType.SGD_PI])
def test_pid_convergence(
    Kp,
    Ki,
    Kd,
    ema_nu,
    pid_init_type,
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

    dual_params = [{"params": _.parameters()} for _ in cmp.multipliers]
    dual_optimizer = PID(dual_params, lr=LR, Kp=Kp, Ki=Ki, Kd=Kd, ema_nu=ema_nu, maximize=True, init_type=pid_init_type)

    cooper_optimizer = cooper.optim.SimultaneousOptimizer(
        primal_optimizers, dual_optimizer, cmp=cmp, multipliers=cmp.multipliers
    )

    for step_id in range(1500):
        cooper_optimizer.roll(compute_cmp_state_kwargs=dict(params=params))

    for param, exact_solution in zip(params, Toy2dCMP_problem_properties["exact_solution"]):
        assert torch.allclose(param, exact_solution)
