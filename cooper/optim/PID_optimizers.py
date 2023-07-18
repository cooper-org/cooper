"""Implementation of a PID controller as a PyTorch optimizer.
Parameters are control variables, and gradients are errors

Inspired by:
https://pytorch-optimizer.readthedocs.io/en/latest/_modules/torch_optimizer/pid.html

Intended to be used on the multipliers, as the P and D terms can reduce oscillations
common to min-max optimization problems.
"""

import warnings
from typing import Optional

import torch

# TODO(juan43ramirez): implement a PID optimizer which incorporates some filtering
# This could be by using EMAs of the error terms


class PID(torch.optim.Optimizer):
    r"""
    Consider a variable x_t, and its gradient g_t = \nabla f(x_t) at time t. This
    optimizer applies PID control to x_t in order to drive g_t to zero and thus find a
    stationary point of f.

    The unrolled PID update is:
        x_{t+1} = x_0 - lr * (Ki * \sum_{i=0}^t g_i + Kp * g_t + Kd * (g_t - g_{t-1}))

    Where Ki is the integral gain, Kp is the proportional gain, and Kd is the derivative
    gain.

    This implementation is *recursive*:
        x_{t+1} = x_t - lr * (Ki * g_t + Kp * (g_t - g_{t-1}) + Kd * (g_t - 2 * g_{t-1} + g_{t-2}))

    Note that setting Ki=1, Kp=0, Kd=0 corresponds to SGD with learning rate lr
    Setting Ki=1, Kp=1, Kd=0 corresponds to the optimistic gradient method

    Arguments:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate
        weight_decay: weight decay (L2 penalty)
        Kp: proportional gain
        Ki: integral gain
        Kd: derivative gain
        maximize: whether to maximize or minimize the loss
    """

    def __init__(
        self,
        params,
        lr: float,
        weight_decay: Optional[float] = 0.0,
        Kp: Optional[float] = 0.0,
        Ki: float = 1.0,
        Kd: Optional[float] = 0.0,
        maximize: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        if Kp < 0.0:
            raise ValueError("Invalid Kp coefficient: {}".format(Kp))
        if Ki < 0.0:
            raise ValueError("Invalid Ki coefficient: {}".format(Ki))
        if Kd < 0.0:
            raise ValueError("Invalid Kd coefficient: {}".format(Kd))
        if all([Kp == 0.0, Ki == 0.0, Kd == 0.0]):
            warnings.warn("Invalid PID coefficients: all are zero")

        defaults = dict(lr=lr, weight_decay=weight_decay, Kp=Kp, Ki=Ki, Kd=Kd, maximize=maximize)

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[float]:
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            Kp, Ki, Kd = group["Kp"], group["Ki"], group["Kd"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                func = _sparse_pid if p.grad.is_sparse else _pid
                func(p, state, lr, weight_decay, Kp, Ki, Kd, maximize)

        return loss


def _estimate_change_and_curvature(grad: torch.Tensor, state: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the change in direction and curvature of the gradient.
        change_t = grad_t - grad_{t-1} = `grad` - `state["previous_direction"]`
        curvature_t = change_t - change_{t-1} = change_t - `state["previous_change"]`
    """

    if len(state) == 0:
        # At this stage, there is not enough information to compute the
        # change in direction for the P term nor the curvature for the D
        # term. Therefore, only the I term is used for the first update.
        change = 0
        curvature = 0
    elif "previous_change" not in state:
        assert "previous_direction" in state
        # Using the previous update direction to compute the P term, but
        # there is not enough information to compute the D term.
        change = grad.sub(state["previous_direction"])
        curvature = 0
    else:
        change = grad.sub(state["previous_direction"])
        curvature = change.sub(state["previous_change"])

    return change, curvature


def _pid(
    param: torch.Tensor, state: dict, lr: float, weight_decay: float, Kp: float, Ki: float, Kd: float, maximize: bool
):
    """
    Applies a PID step update to `param`

    The general form of the update (for minimization, without weight decay) is:
        change_t = grad_t - grad_{t-1}
        curvature_t = change_t - change_{t-1} = grad_t - 2 * grad_{t-1} + grad_{t-2}
        x_{t+1} = x_t - lr * (Ki * grad_t + Kp * change_t + Kd * curvature_t)

    Note that there is not enough information to compute change_0, curvature_0, nor
    curvature_1. Therefore, we define the following convention:
        change_0 = curvature_0 = curvature_1 = 0

    This means that the first update corresponds to SGD with learning rate lr * Ki:
        x_1 = x_0 - lr * Ki * grad_0
    And the second update corresponds to the optimistic gradient method:
        x_2 = x_1 - lr * (Ki * grad_1 + Kp * (grad_1 - grad_0))
    """

    grad = param.grad
    assert not grad.is_sparse

    change, curvature = _estimate_change_and_curvature(grad, state)

    d_p = grad.mul(Ki)

    if Kp != 0:
        d_p.add_(change, alpha=Kp)
    if Kd != 0:
        d_p.add_(curvature, alpha=Kd)

    # Weight decay is applied after estimating the change and curvature, similar to
    # AdamW. See https://arxiv.org/abs/1711.05101 for details.
    if weight_decay != 0:
        d_p.add_(param, alpha=weight_decay)

    if maximize:
        d_p.mul_(-1)

    param.add_(d_p, alpha=-lr)

    if len(state) == 0:
        # Lazily initialize the state after the first update.
        #
        # Only the I term was used for the first update. For the next step,
        # the current update direction will be used to compute the P term.
        # We do not initialize `previous_change` as a convention to indicate
        # that the D term should not be used in the following update.
        state["previous_direction"] = grad.clone().detach()
    else:
        state["previous_direction"] = grad.clone().detach()
        state["previous_change"] = change.clone().detach()


def _sparse_pid(
    param: torch.Tensor, state: dict, lr: float, weight_decay: float, Kp: float, Ki: float, Kd: float, maximize: bool
):
    r"""
    Analogous to _pid but with support for sparse gradients. Inspired by SparseAdam.
    https://github.com/pytorch/pytorch/blob/release/2.0/torch/optim/_functional.py
    """
    grad = param.grad
    assert grad.is_sparse

    grad = grad.coalesce()  # the update is non-linear so indices must be unique
    grad_indices = grad._indices()
    grad_values = grad._values()
    if grad_values.numel() == 0:
        # Skip update for empty grad
        return
    size = grad.size()

    def make_sparse(values):
        """
        Function to convert a dense tensor of values to a sparse representation with the
        same indices as `grad`.
        """

        constructor = grad.new
        if grad_indices.dim() == 0 or values.dim() == 0:
            return constructor().resize_as_(grad)
        return constructor(grad_indices, values, size)

    if len(state) == 0:
        # NOTE: considering a *dense* state. Note that IndexedMultipliers are
        # stored in a dense representation as well.
        state["steps"] = torch.zeros_like(param, dtype=torch.int)
        state["previous_direction"] = torch.zeros_like(param)
        state["previous_change"] = torch.zeros_like(param)

    step_counter = state["steps"].sparse_mask(grad)
    previous_direction = state["previous_direction"].sparse_mask(grad)
    previous_change = state["previous_change"].sparse_mask(grad)

    # Given the available information from previous updates, the change and
    # curvature are estimated or not.
    is_after_first_update = step_counter._values().ge(1)
    is_after_second_update = step_counter._values().ge(2)

    change_values = grad_values.sub(previous_direction._values()).mul(is_after_first_update.float())
    curvature_values = change_values.sub(previous_change._values()).mul(is_after_second_update.float())

    d_p_values = grad_values.mul(Ki)

    if Kp != 0:
        d_p_values.add_(change_values, alpha=Kp)
    if Kd != 0:
        d_p_values.add_(curvature_values, alpha=Kd)

    # Weight decay is applied after estimating the change and curvsture, similar to
    # AdamW. See https://arxiv.org/abs/1711.05101 for details.
    if weight_decay != 0:
        p_values = torch.index_select(param, 0, grad_indices[0])
        d_p_values.add_(p_values, alpha=weight_decay)

    if maximize:
        d_p_values.mul_(-1)

    param.add_(make_sparse(d_p_values), alpha=-lr)

    # Update the step counter for observed parameters.
    state["steps"].add_(make_sparse(torch.ones_like(grad_values, dtype=torch.int)))

    # Update the previous direction and change for observed parameters. We
    # always store `previous_direction` for the next update. `previous_change`
    # is only used for the second update, so we store it using ``
    state["previous_direction"][grad_indices] = grad_values.clone().detach()
    state["previous_change"][grad_indices] = change_values.mul(is_after_first_update.float()).clone().detach()
