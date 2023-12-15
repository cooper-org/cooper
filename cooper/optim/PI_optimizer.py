import warnings
from typing import Optional

import torch


class PI(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        weight_decay: Optional[float] = 0.0,
        Kp: Optional[float] = 0.0,
        Ki: float = 1.0,
        maximize: bool = False,
    ):
        r"""
        Implements a PI controller as a PyTorch optimizer.

        The error signal used for the PI controller is the gradient of a cost function
        :math:`L` being optimized, with parameter :math:`\theta`. We treat :math:`\theta`
        as the control variable, and the gradient of :math:`L` as the error signal. The
        error signal at time :math:`t` is :math:`e_t = \nabla L_t(\theta_t)`. Note that
        the function :math:`L_t` may change over time.

        When ``maximize=False``, the parameter update is multiplied by :math:`-1` before
        being applied.

        The execution of the PI controller is given by:

        .. math::
            \theta_{t+1} &= \theta_t - \text{lr} (K_P (e_t - e_{t-1}) + K_I e_t),

        where :math:`K_P`, :math:`K_I` are the proportional and integral gains,
        respectively. We keep the learning rate :math:`\text{lr}` as a separate
        parameter to facilitate comparison with other optimizers.

        .. note::
            Setting :math:`K_P=0`, :math:`K_I=1`, corresponds to SGD with learning rate
            :math:`\text{lr}`.

            Setting :math:`K_P=1`, :math:`K_I=1` corresponds to the optimistic gradient
            method.

        .. warning::
            This class implements an initialization scheme :math:`e_{-1} = e_{0}`. This
            choice ensures that the first step taken by the optimizer in each direction
            is the same as that of SGD with a learning rate of :math:`\text{lr} K_I`.
            This initialization scheme is not the same as that of :class:`PID`, which
            uses :math:`e_{-1} = 0`.

        Arguments:
            params: iterable of parameters to optimize or dicts defining parameter groups
            lr: learning rate
            weight_decay: weight decay (L2 penalty)
            Kp: proportional gain
            Ki: integral gain
            maximize: whether to maximize or minimize the loss
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        if Kp < 0.0:
            warnings.warn("Using a negative Kp coefficient: {}".format(Kp))
        if Ki < 0.0:
            warnings.warn("Using a negative Ki coefficient: {}".format(Kp))
        if all([Kp == 0.0, Ki == 0.0]):
            warnings.warn("All PI coefficients are zero")

        defaults = dict(lr=lr, weight_decay=weight_decay, Kp=Kp, Ki=Ki, maximize=maximize)

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

                update_function = _sparse_pi if p.grad.is_sparse else _pi
                update_function(
                    param=p,
                    state=self.state[p],
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    Kp=group["Kp"],
                    Ki=group["Ki"],
                    maximize=group["maximize"],
                )

        return loss


def _pi(
    param: torch.Tensor,
    state: dict,
    lr: float,
    weight_decay: float,
    Kp: float,
    Ki: float,
    maximize: bool,
):
    """Applies a PI step update to `param`"""

    error = param.grad
    assert not error.is_sparse, "For sparse updates, use _sparse_pi instead"

    if "previous_error" not in state and Kp != 0:
        # Note that technically, one should initialize the previous error to be
        # :math:`e_{-1} = - (Kp/Ki) e_{0}`. However, since this function implements the
        # recursive formulation of the PI controller
        # :math:`\theta_{t+1} = \theta_t + \text{lr} (K_P (e_t - e_{t-1}) + K_I e_t)`,
        # we can initialize :math:`e_{-1} = e_{0}` to ensure that the first step taken
        # matches that of SGD with learning rate :math:`\text{lr} K_I`.
        state["previous_error"] = error.clone().detach()

    pid_update = torch.zeros_like(param)

    if Ki != 0:
        pid_update.add_(error, alpha=Ki)

    if Kp != 0:
        error_change = error.sub(state["previous_error"])
        pid_update.add_(error_change, alpha=Kp)

    # Weight decay is applied after estimating the delta and curvature, similar to
    # AdamW. See https://arxiv.org/abs/1711.05101 for details.
    if weight_decay != 0:
        pid_update.add_(param, alpha=weight_decay)

    alpha = lr if maximize else -lr
    param.add_(pid_update, alpha=alpha)

    if "previous_error" in state:
        state["previous_error"] = error.clone().detach()


def _sparse_pi(
    param: torch.Tensor,
    state: dict,
    lr: float,
    weight_decay: float,
    Kp: float,
    Ki: float,
    maximize: bool,
):
    """
    Analogous to _pi but with support for sparse gradients.
    """

    error = param.grad
    assert error.is_sparse, "For dense updates, use _pi instead"

    error = error.coalesce()  # the update is non-linear so indices must be unique
    error_indices = error.indices()
    detached_error_values = error._values().clone().detach()

    if detached_error_values.numel() == 0:
        # Skip update for empty grad
        return

    if Kp != 0:
        if "previous_error" not in state:
            state["all_previous_error_initialized"] = False
            state["needs_initialization_mask"] = torch.ones_like(param, dtype=torch.bool)
            state["previous_error"] = torch.zeros_like(param)

        if not state["all_previous_error_initialized"]:
            # Check if current seen indices has been initialized before, if not,
            # then mark as initialized and store first error value in `previous_error`
            state["previous_error"] += error * state["needs_initialization_mask"]
            state["needs_initialization_mask"][error_indices] *= False

            if not torch.any(state["needs_initialization_mask"]):
                # If all buffer entries have been initialized, we flag this so we can
                # cheaply skip this check in the future.
                state["all_previous_error_initialized"] = True

    pid_update_values = torch.zeros_like(detached_error_values)

    if Ki != 0:
        pid_update_values.add_(detached_error_values, alpha=Ki)

    if Kp != 0:
        previous_error = state["previous_error"].sparse_mask(error)
        error_change_values = detached_error_values - previous_error._values()
        pid_update_values.add_(error_change_values, alpha=Kp)

    pid_update = torch.sparse_coo_tensor(error_indices, pid_update_values, size=param.shape)

    # Weight decay is applied after estimating the delta and curvature, similar to
    # AdamW. See https://arxiv.org/abs/1711.05101 for details.
    if weight_decay != 0:
        observed_params = param.sparse_mask(error)
        pid_update.add_(observed_params, alpha=weight_decay)

    alpha = lr if maximize else -lr
    param.add_(pid_update, alpha=alpha)

    if "previous_error" in state:
        state["previous_error"][error_indices] = detached_error_values
