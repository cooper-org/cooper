import cooper_test_utils
import pytest
import torch

import cooper
from cooper.multipliers import IndexedMultiplier
from cooper.optim import PI

# TODO(juan43ramirez): test with multiple parameter groups

ALL_HYPER_PARAMS = [
    (0, 1),  # I controller
    (1, 1),  # PI controller
    (1, 0),  # P controller
]


@pytest.mark.parametrize(["Kp", "Ki"], ALL_HYPER_PARAMS)
@pytest.mark.parametrize("maximize", [True, False])
def test_manual_pi_update(Kp, Ki, maximize):
    param = torch.tensor([1.0], requires_grad=True)

    loss_fn = lambda param: param**2 / 2

    def compute_analytic_gradient():
        # For the quadratic loss, the gradient is simply the current value of p.
        return param.clone().detach()

    def recursive_PI_direction(error, error_change):
        return Kp * error_change + Ki * error

    LR = 0.1
    update_sign = 1 if maximize else -1
    optimizer = PI([param], lr=LR, Kp=Kp, Ki=Ki, maximize=maximize)

    def do_optimizer_step():
        optimizer.zero_grad()
        loss = loss_fn(param)
        loss.backward()
        optimizer.step()

    # Initialization of PI hyperparameters
    error_minus_1 = compute_analytic_gradient()
    previous_error = error_minus_1
    previous_param = param.clone().detach()

    for step_id in range(10):
        new_error = compute_analytic_gradient()
        error_change = new_error - previous_error
        new_param = previous_param + update_sign * LR * recursive_PI_direction(new_error, error_change).clone().detach()

        do_optimizer_step()

        # Check that iterates match
        assert torch.allclose(param, new_param)

        # When entering the next iteration, the "current" error and delta become the "previous"
        if Kp != 0:
            assert torch.allclose(optimizer.state[param]["previous_error"], new_error)

        previous_error = new_error
        previous_param = new_param


@pytest.mark.parametrize(["Kp", "Ki"], ALL_HYPER_PARAMS)
@pytest.mark.parametrize("maximize", [True, False])
def test_manual_sparse_pi_update(Kp, Ki, maximize, device):
    num_multipliers = 10
    multiplier_init = torch.ones(num_multipliers, 1, device=device)
    multiplier_module = IndexedMultiplier(init=multiplier_init, constraint_type=cooper.ConstraintType.EQUALITY)
    param = multiplier_module.weight

    def loss_fn(indices):
        return multiplier_module(indices).pow(2).sum() / 2

    LR = 0.1
    update_sign = 1 if maximize else -1

    def compute_analytic_gradient(indices):
        # For the quadratic loss, the gradient is simply the current value of p.
        return multiplier_module(indices).reshape(-1, 1).clone().detach()

    def recursive_PI_direction(error, error_change):
        return Kp * error_change + Ki * error

    optimizer = PI(multiplier_module.parameters(), lr=LR, Kp=Kp, Ki=Ki, maximize=maximize)

    def do_optimizer_step(indices):
        optimizer.zero_grad()
        loss = loss_fn(indices)
        loss.backward()
        optimizer.step()

    # Initialization of PI hyperparameters
    prev_error_buffer = torch.zeros_like(param)

    # ------------------------ First step ------------------------
    selected_indices_0 = torch.tensor([0, 1, 2, 3], device=device)
    selected_indices_mask_0 = torch.zeros_like(param, dtype=bool)
    selected_indices_mask_0[selected_indices_0] = True
    error_minus_1 = compute_analytic_gradient(selected_indices_0)

    error_0 = compute_analytic_gradient(selected_indices_0)
    prev_error_buffer[selected_indices_0] = error_0.clone().detach()

    new_param = param.clone().detach()
    pi_update = update_sign * LR * recursive_PI_direction(error_0, error_0 - error_minus_1)
    new_param[selected_indices_0] += pi_update.clone().detach()

    do_optimizer_step(selected_indices_0)

    assert torch.allclose(param, new_param)

    if Kp != 0:
        buffer = optimizer.state[param]["previous_error"]
        # Check that entries in modified and unmodified indices have the right values
        assert torch.allclose(buffer[selected_indices_mask_0], error_0)
        assert torch.allclose(buffer[~selected_indices_mask_0], torch.zeros_like(buffer[~selected_indices_mask_0]))

    # ------------------------ Second step ------------------------
    # Note that index 3 was already seen in the previous step, so we don't have to
    # initialize its buffer entry. Indices 4, 8, 9 are unseen.
    selected_indices_1 = torch.tensor([3, 5, 6, 7], device=device)
    selected_indices_mask_1 = torch.zeros_like(param, dtype=bool)
    selected_indices_mask_1[selected_indices_1] = True

    error_1 = compute_analytic_gradient(selected_indices_1)
    # Only the new indices 5, 6, 7, should be initialized with the first seen value
    prev_error_buffer[selected_indices_1[1:]] = error_1.clone().detach()[1:]

    new_param = param.clone().detach()
    pi_update = update_sign * LR * recursive_PI_direction(error_1, error_1 - prev_error_buffer[selected_indices_1])
    new_param[selected_indices_1] += pi_update.clone().detach()

    do_optimizer_step(selected_indices_1)

    assert torch.allclose(param, new_param)

    if Kp != 0:
        buffer = optimizer.state[param]["previous_error"]
        # Check that entries in modified and unmodified indices have the right values
        assert torch.allclose(buffer[selected_indices_mask_0][:-1], error_0)
        assert torch.allclose(buffer[selected_indices_mask_1].reshape(error_1.shape), error_1)
        unseen_indices = torch.tensor([4, 8, 9], device=device)
        assert torch.allclose(buffer[unseen_indices], torch.zeros_like(buffer[unseen_indices]))


ALL_HYPER_PARAMS_CONVERGENCE = [
    (0, 1),  # I controller
    (1, 1),  # PI controller
    (1, 5),  # PI controller
]


@pytest.mark.parametrize(["Kp", "Ki"], ALL_HYPER_PARAMS_CONVERGENCE)
def test_pi_convergence(
    Kp, Ki, Toy2dCMP_problem_properties, Toy2dCMP_params_init, use_multiple_primal_optimizers, device
):
    """Test convergence of PI updates on toy 2D problem. The PI updates are only
    applied to the dual variables."""

    use_ineq_constraints = Toy2dCMP_problem_properties["use_ineq_constraints"]
    if not use_ineq_constraints:
        pytest.skip("Test requires a problem with constraints.")

    cmp = cooper_test_utils.Toy2dCMP(use_ineq_constraints=use_ineq_constraints, device=device)

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=use_multiple_primal_optimizers, params_init=Toy2dCMP_params_init
    )

    dual_params = [{"params": _.parameters()} for _ in cmp.multipliers]
    dual_optimizer = PI(dual_params, lr=0.01, Kp=Kp, Ki=Ki, maximize=True)

    cooper_optimizer = cooper.optim.SimultaneousOptimizer(
        primal_optimizers, dual_optimizer, multipliers=cmp.multipliers
    )

    compute_cmp_state_fn = lambda: cmp.compute_cmp_state(params)

    for step_id in range(1500):
        cooper_optimizer.roll(compute_cmp_state_fn)

    for param, exact_solution in zip(params, Toy2dCMP_problem_properties["exact_solution"]):
        assert torch.allclose(param, exact_solution)
