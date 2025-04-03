import pytest
import torch

import cooper
from cooper.optim import nuPI, nuPIInitType

# TODO(juan43ramirez): test with multiple parameter groups


# Parameter order is: Kp, Ki, ema_nu
ALL_HYPER_PARAMS = [
    (0, 1, 0),  # I controller
    (1, 1, 0),  # PI controller
    (1, 1, 0.9),  # nuPI controller
]

LR = 0.01


@pytest.fixture(params=[1e-2])
def lr(request):
    return request.param


@pytest.fixture(params=[1.0])
def Kp(request):
    return request.param


@pytest.fixture(params=[1.0])
def Ki(request):
    return request.param


@pytest.fixture(params=[0.9])
def ema_nu(request):
    return request.param


@pytest.fixture
def params(device):
    return torch.tensor([1.0, 0.5, 0.8, 1.0], device=device)


def loss_func(params):
    return torch.sum(params**2) / 2.0


def test_i_optimizer(params, lr, Ki):
    """Verify that we can recover GD with LR = Ki * LR whenever Kp=0 and ema_nu=0."""
    params = params.clone().detach().requires_grad_(True)
    params_for_manual_update = params.clone().detach()
    optimizer = nuPI(
        params=[params],
        lr=lr,
        Ki=Ki,
        Kp=0.0,
        ema_nu=0.0,
        weight_decay=0.0,
        maximize=False,
        init_type=nuPIInitType.ZEROS,
    )

    for _ in range(100):
        optimizer.zero_grad()
        loss = loss_func(params)
        loss.backward()

        # nuPI optimizer update
        optimizer.step()

        # Manual update
        manual_grad = params_for_manual_update.clone().detach()
        params_for_manual_update = params_for_manual_update - lr * Ki * manual_grad

        assert torch.allclose(params, params_for_manual_update)


def test_pi_optimizer(params, lr, Kp, Ki):
    """Verify that we can recover PI with whenever ema_nu=0."""
    params = params.clone().detach().requires_grad_(True)
    params_for_manual_update = params.clone().detach()
    optimizer = nuPI(
        params=[params], lr=lr, Ki=Ki, Kp=Kp, ema_nu=0.0, weight_decay=0.0, maximize=False, init_type=nuPIInitType.ZEROS
    )

    previous_grad = torch.zeros_like(params)

    for _ in range(100):
        optimizer.zero_grad()
        loss = loss_func(params)
        loss.backward()

        # nuPI optimizer update
        optimizer.step()

        # Manual update
        manual_grad = params_for_manual_update.clone().detach()  # L2 loss has the parameter as the gradient
        grad_diff = manual_grad - previous_grad
        params_for_manual_update = params_for_manual_update - lr * (Ki * manual_grad + Kp * grad_diff)

        assert torch.allclose(params, params_for_manual_update)
        previous_grad = manual_grad.clone().detach()


def test_manual_nupi_dense_update():
    def loss(param):
        return param.pow(2).sum() / 2

    LR = 1e-2
    KI = 1.0
    KP = torch.tensor([1.0, 0.0, 2.0])
    EMA_NU = 0.5

    parameter = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    optimizer = nuPI([parameter], lr=LR, Kp=KP, Ki=KI, ema_nu=EMA_NU, maximize=False, init_type=nuPIInitType.ZEROS)

    optimizer.zero_grad()
    loss(parameter).backward()

    xi_m1 = torch.zeros_like(parameter)
    update_1 = (KI + KP * (1 - EMA_NU)) * parameter.grad + EMA_NU * KP * xi_m1
    expected_param_1 = parameter.clone() - LR * update_1

    optimizer.step()
    assert torch.allclose(parameter, expected_param_1)

    # -------------------- 2nd step --------------------
    # Compute optimizer state variable with previous gradient (before zero_grad)
    xi_0 = EMA_NU * xi_m1 + (1 - EMA_NU) * parameter.grad

    optimizer.zero_grad()
    loss(parameter).backward()

    update_2 = (KI + KP * (1 - EMA_NU)) * parameter.grad - KP * (1 - EMA_NU) * xi_0
    expected_param_2 = parameter.clone() - LR * update_2

    optimizer.step()
    assert torch.allclose(parameter, expected_param_2)


@pytest.mark.parametrize(("Kp", "Ki", "ema_nu"), ALL_HYPER_PARAMS)
@pytest.mark.parametrize("maximize", [True, False])
def test_sparse_nupi_update_zeros_init(Kp, Ki, ema_nu, maximize, device):
    num_multipliers = 10
    multiplier_init = torch.ones(num_multipliers, device=device)
    multiplier_module = cooper.multipliers.IndexedMultiplier(init=multiplier_init)
    param = multiplier_module.weight

    def loss_fn(indices):
        return multiplier_module(indices).pow(2).sum() / 2

    update_sign = 1 if maximize else -1

    def compute_analytic_gradient(indices):
        # For the quadratic loss, the gradient is simply the current value of p.
        return multiplier_module(indices).reshape(-1, 1).clone().detach()

    def recursive_nuPI_direction(error, previous_xi):
        return (Ki + (1 - ema_nu) * Kp) * error - (1 - ema_nu) * Kp * previous_xi

    optimizer = nuPI(
        multiplier_module.parameters(),
        lr=LR,
        Kp=Kp,
        Ki=Ki,
        ema_nu=ema_nu,
        maximize=maximize,
        init_type=nuPIInitType.ZEROS,
    )

    def do_optimizer_step(indices):
        optimizer.zero_grad()
        loss = loss_fn(indices)
        loss.backward()
        optimizer.step()

    # Initialization of PID hyperparameters
    previous_xi_buffer = torch.zeros_like(param)

    # ------------------------ First step ------------------------
    selected_indices_0 = torch.tensor([0, 1, 2, 3], device=device)
    selected_indices_mask_0 = torch.zeros_like(param, dtype=bool)
    selected_indices_mask_0[selected_indices_0] = True

    error_0 = compute_analytic_gradient(selected_indices_0)
    xi_0 = ema_nu * previous_xi_buffer[selected_indices_0] + (1 - ema_nu) * error_0
    previous_xi_values = previous_xi_buffer[selected_indices_0].clone().detach()

    new_param = param.clone().detach()
    nupi_update = update_sign * LR * recursive_nuPI_direction(error_0, previous_xi_values)
    new_param[selected_indices_0] += nupi_update.clone().detach()

    do_optimizer_step(selected_indices_0)
    assert torch.allclose(param, new_param)

    previous_xi_buffer[selected_indices_0] = xi_0.clone().detach()

    if Kp != 0:
        buffer = optimizer.state[param]["xi"]
        # Check that entries in modified and unmodified indices have the right values
        assert torch.allclose(buffer[selected_indices_mask_0], xi_0)
        assert torch.allclose(buffer[~selected_indices_mask_0], torch.zeros_like(buffer[~selected_indices_mask_0]))

    # # ------------------------ Second step ------------------------
    # Note that index 3 was already seen in the previous step, so we don't have to
    # initialize its buffer entry. Indices 4, 8, 9 are unseen.
    selected_indices_1 = torch.tensor([3, 5, 6, 7], device=device)
    selected_indices_mask_1 = torch.zeros_like(param, dtype=bool)
    selected_indices_mask_1[selected_indices_1] = True

    error_1 = compute_analytic_gradient(selected_indices_1)
    xi_1 = ema_nu * previous_xi_buffer[selected_indices_1] + (1 - ema_nu) * error_1
    previous_xi_values = previous_xi_buffer[selected_indices_1].clone().detach()

    new_param = param.clone().detach()
    nupi_update = update_sign * LR * recursive_nuPI_direction(error_1, previous_xi_values)
    new_param[selected_indices_1] += nupi_update.clone().detach()

    do_optimizer_step(selected_indices_1)
    assert torch.allclose(param, new_param)

    previous_xi_buffer[selected_indices_1] = xi_1.clone().detach()

    if Kp != 0:
        buffer = optimizer.state[param]["xi"]
        # Check that entries in modified and unmodified indices have the right values
        assert torch.allclose(buffer[selected_indices_mask_0][:-1], xi_0[:-1])  # Skip index 3
        # Check state entries updated in this iteration
        assert torch.allclose(buffer[selected_indices_mask_1].reshape(xi_1.shape), xi_1)
        # Check state entries that have not been updated yet
        unseen_indices = torch.tensor([4, 8, 9], device=device)
        assert torch.allclose(buffer[unseen_indices], torch.zeros_like(buffer[unseen_indices]))


def test_nupi_sgd_init_matches_sgd(params, lr, Kp, Ki):
    """Verify that the first step of nuPI is equivalent to PI whenever ema_nu=0."""
    params = params.clone().detach().requires_grad_(True)
    params_for_manual_update = params.clone().detach()
    optimizer = nuPI(
        params=[params], lr=lr, Ki=Ki, Kp=Kp, ema_nu=0.0, weight_decay=0.0, maximize=False, init_type=nuPIInitType.SGD
    )

    optimizer.zero_grad()
    loss = loss_func(params)
    loss.backward()

    # nuPI optimizer update
    optimizer.step()

    # Manual update
    manual_grad = params_for_manual_update.clone().detach()  # L2 loss has the parameter as the gradient
    params_for_manual_update = params_for_manual_update - lr * Ki * manual_grad

    assert torch.allclose(params, params_for_manual_update)


@pytest.mark.parametrize(("Kp", "Ki", "ema_nu"), ALL_HYPER_PARAMS)
@pytest.mark.parametrize("maximize", [True, False])
def test_sparse_nupi_update_sgd_init(Kp, Ki, ema_nu, maximize, device):
    """Verify the behavior of nuPI when initialized with SGD."""
    num_multipliers = 10
    multiplier_init = torch.ones(num_multipliers, device=device)
    multiplier_module = cooper.multipliers.IndexedMultiplier(init=multiplier_init)
    param = multiplier_module.weight

    def loss_fn(indices):
        return multiplier_module(indices).pow(2).sum() / 2

    update_sign = 1 if maximize else -1

    def compute_analytic_gradient(indices):
        # For the quadratic loss, the gradient is simply the current value of p.
        return multiplier_module(indices).reshape(-1, 1).clone().detach()

    optimizer = nuPI(
        multiplier_module.parameters(),
        lr=LR,
        Kp=Kp,
        Ki=Ki,
        ema_nu=ema_nu,
        maximize=maximize,
        init_type=nuPIInitType.SGD,
    )

    def do_optimizer_step(indices):
        optimizer.zero_grad()
        loss = loss_fn(indices)
        loss.backward()
        optimizer.step()

    # Initialization of PID hyperparameters
    xi_buffer = torch.zeros_like(param)

    # ------------------------ First step ------------------------
    selected_indices_0 = torch.tensor([0, 1, 2, 3], device=device)
    selected_indices_mask_0 = torch.zeros_like(param, dtype=bool)
    selected_indices_mask_0[selected_indices_0] = True

    error_0 = compute_analytic_gradient(selected_indices_0)

    new_param = param.clone().detach()
    nupi_update = update_sign * LR * Ki * error_0  # First step should match GD
    new_param[selected_indices_0] += nupi_update.clone().detach()

    do_optimizer_step(selected_indices_0)
    assert torch.allclose(param, new_param)

    xi_1_values = ema_nu * xi_buffer[selected_indices_0] + (1 - ema_nu) * error_0
    xi_buffer[selected_indices_0] = xi_1_values.clone().detach()

    if Kp != 0:
        buffer = optimizer.state[param]["xi"]
        # Check that entries in modified and unmodified indices have the right values
        assert torch.allclose(buffer[selected_indices_mask_0], xi_1_values)
        assert torch.allclose(buffer[~selected_indices_mask_0], torch.zeros_like(buffer[~selected_indices_mask_0]))

    # # ------------------------ Second step ------------------------
    # Note that index 3 was already seen in the previous step, so we don't have to
    # initialize its buffer entry. Indices 4, 8, 9 are unseen.
    selected_indices_1 = torch.tensor([3, 5, 6, 7], device=device)
    selected_indices_mask_1 = torch.zeros_like(param, dtype=bool)
    selected_indices_mask_1[selected_indices_1] = True

    error_1 = compute_analytic_gradient(selected_indices_1)

    new_param = param.clone().detach()
    # Index 3 is the only one that has been seen before, so we apply the recursive update
    nupi_direction_ix_0 = Ki * error_1[0] + Kp * (1 - ema_nu) * (error_1[0] - xi_buffer[selected_indices_1[0]])
    nupi_update_ix_0 = update_sign * LR * nupi_direction_ix_0
    new_param[selected_indices_1[0]] += nupi_update_ix_0.clone().detach()
    # Indices 5, 6, 7 have not been seen before, so we apply the SGD update
    nupi_update_ix_1 = update_sign * LR * Ki * error_1[1:]
    new_param[selected_indices_1[1:]] += nupi_update_ix_1.clone().detach()

    do_optimizer_step(selected_indices_1)
    assert torch.allclose(param, new_param)

    xi_2_values = ema_nu * xi_buffer[selected_indices_1] + (1 - ema_nu) * error_1
    xi_buffer[selected_indices_1] = xi_2_values.clone().detach()

    if Kp != 0:
        buffer = optimizer.state[param]["xi"]
        # Check that entries in modified and unmodified indices have the right values
        assert torch.allclose(buffer[selected_indices_mask_0][:-1], xi_1_values[:-1])  # Skip index 3
        # Check state entries updated in this iteration
        assert torch.allclose(buffer[selected_indices_mask_1].reshape(xi_2_values.shape), xi_2_values)
        # Check state entries that have not been updated yet
        unseen_indices = torch.tensor([4, 8, 9], device=device)
        assert torch.allclose(buffer[unseen_indices], torch.zeros_like(buffer[unseen_indices]))
