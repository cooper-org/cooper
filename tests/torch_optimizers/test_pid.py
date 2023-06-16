import cooper_test_utils
import pytest
import torch

import cooper
from cooper.multipliers import IndexedMultiplier
from cooper.optim import PID, SparsePID


@pytest.mark.parametrize(["proportional", "integral", "derivative"], [(0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1)])
def test_manual_pid(proportional, integral, derivative):
    param = torch.tensor([1.0], requires_grad=True)

    loss_fn = lambda param: param**2 / 2

    lr = 0.1

    def PID_direction(grad, change, curvature):
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
    assert "previous_change" not in optimizer.state[param]

    # -------------------------------------------- Second step of optimization
    p2 = p1 - lr * PID_direction(p1, p1 - p0, 0.0)

    do_step()

    assert torch.allclose(param, p2)

    # The state contain p1 in `previous_direction` and p1 - p0 in `previous_change`.
    assert torch.allclose(optimizer.state[param]["previous_direction"], p1)
    assert torch.allclose(optimizer.state[param]["previous_change"], p1 - p0)

    # -------------------------------------------- Third step of optimization
    p3 = p2 - lr * PID_direction(p2, p2 - p1, p2 - 2 * p1 + p0)

    do_step()

    assert torch.allclose(param, p3)

    # The state contain p2 in `previous_direction` and p2 - p1 in `previous_change`.
    assert torch.allclose(optimizer.state[param]["previous_direction"], p2)
    assert torch.allclose(optimizer.state[param]["previous_change"], p2 - p1)


@pytest.mark.parametrize(["proportional", "integral", "derivative"], [(0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1)])
def test_manual_sparse_pid(proportional, integral, derivative):
    if not torch.cuda.is_available():
        pytest.skip("Sparse gradients are only supported on GPU.")

    device = "cuda"

    param = IndexedMultiplier(torch.ones(10, 1, device=device), use_sparse_gradient=True)
    all_indices = torch.arange(10, device=device)

    def loss_fn(param, index):
        return param(index).pow(2).sum() / 2

    lr = 0.1

    def PID_direction(grad, change, curvature):
        return integral * grad + proportional * change + derivative * curvature

    optimizer = SparsePID(
        param.parameters(), lr=lr, proportional=proportional, integral=integral, derivative=derivative
    )

    # -------------------------------------------- First step of optimization
    indices = torch.tensor([0, 1, 4, 7], device=device)

    # Manual first step of optimization. The gradient is simply the current value of p.
    p0 = param(all_indices).clone().detach()
    p1 = p0.clone()
    p1[indices] -= lr * PID_direction(p0, 0.0, 0.0)[indices]  # only update the current indices

    # Manually calculate the state of the optimizer
    previous_direction = torch.zeros_like(p0)
    previous_direction[indices] = p0.clone()[indices]
    # For the first step, the previous_change is 0 for all indices
    previous_change = torch.zeros_like(p0)

    optimizer.zero_grad()
    loss = loss_fn(param, indices)
    loss.backward()
    optimizer.step()

    assert torch.allclose(param(all_indices), p1)

    # Check the state of the optimizer. For observer indices, should contain the first
    # gradient in `previous_direction` and also in `previous_change`. 0 otherwise.
    state = optimizer.state[list(param.parameters())[0]]

    assert torch.allclose(state["previous_direction"].flatten(), previous_direction)
    assert torch.allclose(state["previous_change"].flatten(), previous_change)

    # -------------------------------------------- Second step of optimization
    old_indices = indices.clone()

    indices = torch.tensor([0, 3, 5, 7, 9], device=device)

    # Manual second step
    p2 = p1.clone()
    change = p1 - previous_direction

    unseen_indices = torch.ones_like(all_indices, dtype=torch.bool)
    unseen_indices[old_indices] = False
    change[unseen_indices] = 0.0

    p2[indices] -= lr * PID_direction(p1, change, 0.0)[indices]

    previous_direction[indices] = p1.clone()[indices]

    # We modify previous_change only for the indices that were updated.
    # For indices that were only observed now, the previous_change is 0
    previous_change[indices] = torch.zeros_like(p1[indices])

    # For indices that were observed before *and* now, the previous_change is p1 - p0.
    # Note that the following line overwites the previous line for the indices that
    # belong to both old_indices and indices.
    twice_observed_indices = [idx for idx in range(10) if idx in old_indices and idx in indices]
    previous_change[twice_observed_indices] = p1[twice_observed_indices] - p0[twice_observed_indices]

    optimizer.zero_grad()
    loss = loss_fn(param, indices)
    loss.backward()
    optimizer.step()

    assert torch.allclose(param(all_indices), p2)

    state = optimizer.state[list(param.parameters())[0]]

    assert torch.allclose(state["previous_direction"].flatten(), previous_direction)
    assert torch.allclose(state["previous_change"].flatten(), previous_change)

    # -------------------------------------------- Third step of optimization


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
