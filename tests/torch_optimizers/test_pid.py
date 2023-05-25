import cooper_test_utils
import pytest
import torch

from cooper.optim import PID, SparsePID

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize(["proportional", "integral", "derivative"], [(0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1)])
def test_manual_pid(proportional, integral, derivative):
    param = torch.tensor([1.0], requires_grad=True)

    def loss_fn(param):
        return param**2 / 2

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
