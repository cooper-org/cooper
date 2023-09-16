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


class PID(torch.optim.Optimizer):
    r"""
    Implements a PID controller as a PyTorch optimizer.

    The error signal used for the PID controller is the gradient of a cost function
    :math:`L` being optimized, with parameter :math:`\theta`. We treat :math:`\theta`
    as the control variable, and the gradient of :math:`L` as the error signal. The
    error signal at time :math:`t` is :math:`e_t = \nabla L_t(\theta_t)`. Note that
    the function :math:`L_t` may change over time.

    When ``maximize=False``, the incoming error signal is multiplied by :math:`-1`.

    The execution of the PID controller is given by:

    .. math::
        \partial_t &= \nu \partial_{t-1} + (1 - \nu) (e_t - e_{t-1}), \\\\
        \theta_{t+1} &= \theta_t - \text{lr} (K_P (e_t - e_{t-1} + K_I e_t + K_D (\partial_t - \partial_{t-1})),

    where :math:`K_P`, :math:`K_I`, and :math:`K_D` are the proportional, integral, and
    derivative gains, respectively, and :math:`\nu` is the EMA coefficient used to
    reduce noise in the estimation of the derivative term.
    We keep the learning rate :math:`\text{lr}` as a separate parameter to facilitate
    comparison with other optimizers.

    .. note::
        :math:`e_{-1}` and :math:`\partial_{-1}` are hyperparameters of the optimizer
        which require initialization. Typically, :math:`e_{-1} = 0` and
        :math:`\partial_{-1} = 0`.

    .. note::
        Setting :math:`K_P=0`, :math:`K_I=1`, and :math:`K_D=0` corresponds to
        SGD with learning rate :math:`\text{lr}`.

        Setting :math:`K_P=1`, :math:`K_I=1`, and :math:`K_D=0` corresponds to the
        optimistic gradient method.

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
        ema_nu: Optional[float] = 0.0,
        maximize: bool = False,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        # We allowing for negative PID coefficients Kp, Ki and Kd to realize common
        # momentum-based methods as instances of PID control.
        if Kp < 0.0:
            warnings.warn("Using a negative Kp coefficient: {}".format(Kp))
        if Ki < 0.0:
            warnings.warn("Using a negative Ki coefficient: {}".format(Kp))
        if Kd < 0.0:
            warnings.warn("Using a negative Kd coefficient: {}".format(Kp))
        if all([Kp == 0.0, Ki == 0.0, Kd == 0.0]):
            warnings.warn("All PID coefficients are zero")

        if ema_nu < 0.0:
            warnings.warn("Using a negative EMA coefficient: {}".format(ema_nu))
        elif ema_nu >= 1.0:
            raise ValueError("EMA coefficient is above one: {}".format(ema_nu))
        if ema_nu != 0.0 and Kd == 0.0:
            warnings.warn("EMA coefficient is non-zero but Kd is zero")

        defaults = dict(lr=lr, weight_decay=weight_decay, Kp=Kp, Ki=Ki, Kd=Kd, ema_nu=ema_nu, maximize=maximize)

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
                    ema_nu=group["ema_nu"],
                    maximize=group["maximize"],
                )

        return loss


def _pid(
    param: torch.Tensor,
    state: dict,
    lr: float,
    weight_decay: float,
    Kp: float,
    Ki: float,
    Kd: float,
    ema_nu: float,
    maximize: bool,
):
    """Applies a PID step update to `param`"""

    error = param.grad
    assert not error.is_sparse, "For sparse updates, use _sparse_pid instead"

    if "previous_error" not in state and (Kp != 0 or Ki != 0):
        state["previous_error"] = torch.zeros_like(error)
    if "previous_delta" not in state and (Kd != 0):
        state["previous_delta"] = torch.zeros_like(error)

    if not maximize:
        error.mul_(-1)

    pid_update = torch.zeros_like(param)

    if Ki != 0:
        pid_update.add_(error, alpha=Ki)

    if Kp != 0 or Kd != 0:
        error_change = error.sub(state["previous_error"])

        if Kp != 0:
            pid_update.add_(error_change, alpha=Kp)

        if Kd != 0:
            new_delta = state["previous_delta"].mul(ema_nu).add(error_change, alpha=1 - ema_nu)
            delta_change = new_delta.sub(state["previous_delta"])
            pid_update.add_(delta_change, alpha=Kd)

    # Weight decay is applied after estimating the delta and curvature, similar to
    # AdamW. See https://arxiv.org/abs/1711.05101 for details.
    if weight_decay != 0:
        weight_decay_sign = -1 if maximize else 1
        pid_update.add_(param, alpha=weight_decay_sign * weight_decay)

    param.add_(pid_update, alpha=lr)

    if "previous_error" in state:
        state["previous_error"] = error.detach()
    if "previous_delta" in state:
        state["previous_delta"] = new_delta.detach()


def _sparse_pid(
    param: torch.Tensor,
    state: dict,
    lr: float,
    weight_decay: float,
    Kp: float,
    Ki: float,
    Kd: float,
    ema_nu: float,
    maximize: bool,
):
    """
    Analogous to _pid but with support for sparse gradients.
    Inspired by SparseAdam:
    https://github.com/pytorch/pytorch/blob/release/2.0/torch/optim/_functional.py
    """

    error = param.grad
    assert error.is_sparse, "For dense updates, use _pid instead"

    error = error.coalesce()  # the update is non-linear so indices must be unique
    error_indices = error._indices()
    error_values = error._values()

    if error_values.numel() == 0:
        # Skip update for empty grad
        return

    if "previous_error" not in state and (Kp != 0 or Ki != 0):
        state["previous_error"] = torch.zeros_like(param)
    if "previous_delta" not in state and (Kd != 0):
        state["previous_delta"] = torch.zeros_like(param)

    if not maximize:
        error.mul_(-1)

    pid_update = torch.zeros_like(param)

    if Ki != 0:
        pid_update.add_(error, alpha=Ki)

    if Kp != 0 or Kd != 0:
        previous_error = state["previous_error"].sparse_mask(error)
        error_change = error.sub(previous_error)

        if Kp != 0:
            pid_update.add_(error_change, alpha=Kp)

        if Kd != 0:
            previous_delta = state["previous_delta"].sparse_mask(error)
            new_delta = previous_delta.mul(ema_nu).add(error_change, alpha=1 - ema_nu)
            delta_change = new_delta.sub(previous_delta)
            pid_update.add_(delta_change, alpha=Kd)

    # Weight decay is applied after estimating the delta and curvature, similar to
    # AdamW. See https://arxiv.org/abs/1711.05101 for details.
    if weight_decay != 0:
        weight_decay_sign = -1 if maximize else 1
        observed_params = param.sparse_mask(error)
        pid_update.add_(observed_params, alpha=weight_decay_sign * weight_decay)

    param.add_(pid_update, alpha=lr)

    if "previous_error" in state:
        state["previous_error"][error_indices] = error._values().detach()
    if "previous_delta" in state:
        state["previous_delta"][error_indices] = new_delta._values().detach()
