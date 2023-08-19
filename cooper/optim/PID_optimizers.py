"""Implementation of a PID controller as a PyTorch optimizer.
The parameters are treated as the control variables, and the gradients are considered
the error signal, which we aim to drive to zero.

Inspired by:
https://pytorch-optimizer.readthedocs.io/en/latest/_modules/torch_optimizer/pid.html

This optimizer is intended to be used on the Lagrange multipliers, as the (P) and (D)
terms can help reduce oscillations common to min-max optimization problems.
"""

import warnings
from typing import Optional

import torch

# TODO(juan43ramirez): current implementation always keeps buffers for calculating the
# momentum, (P), and (D) terms. Some of these are not necessary when the momentum, Kp,
# or Kd coefficients are zero. This could be optimized to save memory.


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
    Setting Ki=1, Kp=1, Kd=0 corresponds to the optimistic gradient method.

    # TODO(juan43ramirez): document the momentum term

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
        momentum: Optional[float] = 0.0,
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
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(lr=lr, weight_decay=weight_decay, Kp=Kp, Ki=Ki, Kd=Kd, momentum=momentum, maximize=maximize)

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
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                update_function = _sparse_pid if p.grad.is_sparse else _pid
                update_function(
                    param=p,
                    state=self.state[p],
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    Kp=group["Kp"],
                    Ki=group["Ki"],
                    Kd=group["Kd"],
                    momentum=group["momentum"],
                    maximize=group["maximize"],
                )

        return loss


def _estimate_deltas_and_curvature(
    param, grad: torch.Tensor, state: dict
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the gradient delta and curvature.
        delta_t = grad_t - grad_{t-1} = `grad` - `state["previous_grad"]`
        curvature_t = delta_t - delta_{t-1} = delta_t - `state["previous_delta"]`
    """

    if len(state) == 0:
        # At this stage, there is not enough information to compute the parameter or
        # gradient delta for the (P) term nor the curvature for the (D) term. Therefore,
        # only the (I) term is used for the first update.
        param_delta = 0
        grad_delta = 0
        curvature = 0
    elif "previous_delta" not in state:
        assert "previous_grad" in state
        # We use the previous grad to compute the momentum and (P) terms, but there is
        # still not enough information to compute the (D) term.
        param_delta = param.sub(state["previous_param"])
        grad_delta = grad.sub(state["previous_grad"])
        curvature = 0
    else:
        param_delta = param.sub(state["previous_param"])
        grad_delta = grad.sub(state["previous_grad"])
        curvature = grad_delta.sub(state["previous_delta"])

    return param_delta, grad_delta, curvature


def _pid(
    param: torch.Tensor,
    state: dict,
    lr: float,
    weight_decay: float,
    Kp: float,
    Ki: float,
    Kd: float,
    momentum: float,
    maximize: bool,
):
    """
    Applies a PID step update to `param`

    The general form of the update (for minimization, without weight decay) is:
        delta_t = grad_t - grad_{t-1}
        curvature_t = grad_t - 2 * grad_{t-1} + grad_{t-2} = delta_t - delta_{t-1}
        x_{t+1} = x_t - lr * (Ki * grad_t + Kp * delta_t + Kd * curvature_t)

    # TODO(juan43ramirez): address the momentum term

    Note that there is not enough information to compute delta_0, curvature_0, nor
    curvature_1. Therefore, we define the following convention:
        delta_0 = curvature_0 = curvature_1 = 0

    This means that the first update corresponds to SGD with learning rate lr * Ki:
        x_1 = x_0 - lr * Ki * grad_0
    And the second update corresponds to the optimistic gradient method:
        x_2 = x_1 - lr * (Ki * grad_1 + Kp * (grad_1 - grad_0))
    """

    grad = param.grad
    assert not grad.is_sparse, "For sparse gradients, use _sparse_pid instead"

    param_delta, grad_delta, curvature = _estimate_deltas_and_curvature(param, grad, state)

    d_p = grad.mul(Ki)

    if momentum != 0:
        d_p.add_(param_delta, alpha=momentum)
    if Kp != 0:
        d_p.add_(grad_delta, alpha=Kp)
    if Kd != 0:
        d_p.add_(curvature, alpha=Kd)

    # Weight decay is applied after estimating the delta and curvature, similar to
    # AdamW. See https://arxiv.org/abs/1711.05101 for details.
    if weight_decay != 0:
        d_p.add_(param, alpha=weight_decay)

    if maximize:
        d_p.mul_(-1)

    if len(state) == 0:
        # Lazily initialize the state after the first update.
        # Only the (I) term was used for the first update. For the next step, the
        # current parameter will be used to compute the momentum term, and the current
        # gradient will be used to compute the (P) term.
        # We do not initialize `previous_delta` as a convention to indicate
        # that the (D) term should not be used in the following update.
        state["previous_param"] = param.clone().detach()
        state["previous_grad"] = grad.clone().detach()
    else:
        state["previous_param"] = param.clone().detach()
        state["previous_grad"] = grad.clone().detach()
        state["previous_delta"] = grad_delta.clone().detach()

    param.add_(d_p, alpha=-lr)


def _sparse_pid(
    param: torch.Tensor,
    state: dict,
    lr: float,
    weight_decay: float,
    Kp: float,
    Ki: float,
    Kd: float,
    momentum: float,
    maximize: bool,
):
    r"""
    Analogous to _pid but with support for sparse gradients. Inspired by SparseAdam.
    https://github.com/pytorch/pytorch/blob/release/2.0/torch/optim/_functional.py
    """
    grad = param.grad
    assert grad.is_sparse, "For dense gradients, use _pid instead"

    grad = grad.coalesce()  # the update is non-linear so indices must be unique
    grad_indices = grad._indices()
    grad_values = grad._values()
    observed_param_values = torch.index_select(param, 0, grad_indices[0])

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
        # NOTE: considering a *dense* state. Note that tensors related to IndexedMultipliers
        # are stored in a dense representation as well.
        state["steps"] = torch.zeros_like(param, dtype=torch.int)
        state["previous_param"] = torch.zeros_like(param)
        state["previous_grad"] = torch.zeros_like(param)
        state["previous_delta"] = torch.zeros_like(param)

    step_counter = state["steps"].sparse_mask(grad)
    # Given the available information from previous updates, the deltas and curvature
    # are estimated or not.
    is_after_first_update = step_counter._values().ge(1)
    is_after_second_update = step_counter._values().ge(2)

    previous_param = state["previous_param"].sparse_mask(grad)
    previous_grad = state["previous_grad"].sparse_mask(grad)
    previous_delta = state["previous_delta"].sparse_mask(grad)

    param_delta_values = observed_param_values.sub(previous_param._values()).mul(is_after_first_update.float())
    delta_values = grad_values.sub(previous_grad._values()).mul(is_after_first_update.float())
    curvature_values = delta_values.sub(previous_delta._values()).mul(is_after_second_update.float())

    d_p_values = grad_values.mul(Ki)

    if momentum != 0:
        d_p_values.add_(param_delta_values, alpha=momentum)
    if Kp != 0:
        d_p_values.add_(delta_values, alpha=Kp)
    if Kd != 0:
        d_p_values.add_(curvature_values, alpha=Kd)

    # Weight decay is applied after estimating the delta and curvature, similar to
    # AdamW. See https://arxiv.org/abs/1711.05101 for details.
    if weight_decay != 0:
        d_p_values.add_(observed_param_values, alpha=weight_decay)

    if maximize:
        d_p_values.mul_(-1)

    # Update the step counter for observed parameters.
    state["steps"].add_(make_sparse(torch.ones_like(grad_values, dtype=torch.int)))

    # Update the previous param, grad, and delta for observed parameters. We always
    # store `previous_grad` for the next update. `previous_delta` is only used for the
    # second update, so we store it only for the parameters that have been observed
    # after their first update.
    state["previous_param"][grad_indices] = observed_param_values.clone().detach()
    state["previous_grad"][grad_indices] = grad_values.clone().detach()
    state["previous_delta"][grad_indices] = delta_values.mul(is_after_first_update.float()).clone().detach()

    param.add_(make_sparse(d_p_values), alpha=-lr)
