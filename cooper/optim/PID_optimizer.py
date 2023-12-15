from enum import Enum
import warnings
from typing import Optional

import torch


class PIDInitType(Enum):
    """
    PID initialization types. This is used to determine how to initialize the
    error and derivative terms of the PID controller. The initialization scheme
    `SGD_PI` ensures that the first step of `PID(KP, KI, KD)` is equivalent to SGD with
    learning rate :math:`\text{lr} K_I`, and that the second step of `PID(KP, KI, KD)`
    matches the second step of `PI(KP, KI, 0)`. Note that the implementation of the `PI`
    optimizer is also designed to match `SGD(lr=lr*KI)` in the first step.
    """

    ZEROS = 0
    SGD_PI = 1


class PID(torch.optim.Optimizer):
    r"""
    Implements a PID controller as a PyTorch optimizer.

    The error signal used for the PID controller is the gradient of a cost function
    :math:`L` being optimized, with parameter :math:`\theta`. We treat :math:`\theta`
    as the control variable, and the gradient of :math:`L` as the error signal. The
    error signal at time :math:`t` is :math:`e_t = \nabla L_t(\theta_t)`. Note that
    the function :math:`L_t` may change over time.

    When ``maximize=False``, the parameter update is multiplied by :math:`-1` before
    being applied.

    The execution of the PID controller is given by:

    .. math::
        \partial_t &= \nu \partial_{t-1} + (1 - \nu) (e_t - e_{t-1}), \\\\
        \theta_{t+1} &= \theta_t - \text{lr} (K_P (e_t - e_{t-1} + K_I e_t + K_D (\partial_t - \partial_{t-1})),

    where :math:`K_P`, :math:`K_I`, and :math:`K_D` are the proportional, integral, and
    derivative gains. :math:`\nu` is the EMA coefficient used to reduce the noise in 
    the estimation of the derivative term. We keep the learning rate :math:`\text{lr}`
    as a separate parameter to facilitate comparison with other optimizers.

    .. note::
        Setting :math:`K_P=0`, :math:`K_I=1`, and :math:`K_D=0` is equivalent to running
        SGD with learning rate :math:`\text{lr}`.

        Setting :math:`K_P=1`, :math:`K_I=1`, and :math:`K_D=0` corresponds to the
        optimistic gradient method.

    .. note::
        :math:`e_{-1}` and :math:`\partial_{-1}` are hyperparameters of the optimizer
        which require initialization. Typically, :math:`e_{-1} = 0` and
        :math:`\partial_{-1} = 0`.

    .. warning::
        This implementation assumes an initialization of :math:`e_{-1} = 0` and
        :math:`\partial_{-1} = 0`. Currently NAG-short is not supported since it would
        require a different initialization of :math:`\partial_{-1}`.

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
        init_type: PIDInitType = PIDInitType.SGD_PI,
        maximize: bool = False,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        if not init_type in [PIDInitType.ZEROS, PIDInitType.SGD_PI]:
            raise ValueError("Invalid init_type: {}".format(init_type))

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

        defaults = dict(
            lr=lr, weight_decay=weight_decay, Kp=Kp, Ki=Ki, Kd=Kd, ema_nu=ema_nu, maximize=maximize, init_type=init_type
        )

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

                if p.grad.is_sparse:
                    if group["init_type"] == PIDInitType.ZEROS:
                        update_function = _sparse_pid_zero_init
                    elif group["init_type"] == PIDInitType.SGD_PI:
                        update_function = _sparse_pid_sgd_pi_init
                else:
                    update_function = _pid

                update_function(
                    param=p,
                    state=self.state[p],
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    Kp=group["Kp"],
                    Ki=group["Ki"],
                    Kd=group["Kd"],
                    ema_nu=group["ema_nu"],
                    init_type=group["init_type"],
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
    init_type: PIDInitType,
    maximize: bool,
):
    """Applies a PID step update to `param`"""

    uses_error_buffer = Kp != 0 or Kd != 0
    uses_delta_buffer = Kd != 0

    error = param.grad
    assert not error.is_sparse, "For sparse updates, use _sparse_pid instead"

    # This case initializes the buffers to zero under the `PIDInitType.ZEROS` scheme.
    # For the `PIDInitType.SGD_PI` scheme, the buffers are initialized below.
    if init_type == PIDInitType.ZEROS:
        if uses_error_buffer and ("previous_error" not in state):
            state["previous_error"] = torch.zeros_like(error)
        if uses_delta_buffer and ("previous_delta" not in state):
            state["previous_delta"] = torch.zeros_like(error)

    new_delta = None
    pid_update = torch.zeros_like(param)

    if Ki != 0:
        pid_update.add_(error, alpha=Ki)

    if "previous_error" in state:
        error_change = error.sub(state["previous_error"])

        if Kp != 0:
            pid_update.add_(error_change, alpha=Kp)

        if Kd != 0:
            if "previous_delta" in state:
                new_delta = state["previous_delta"].mul(ema_nu).add(error_change, alpha=1 - ema_nu)
                delta_change = new_delta.sub(state["previous_delta"])
                pid_update.add_(delta_change, alpha=Kd)
            else:
                # First time delta is populated, we use the first valid error change
                # This branch handles the initializtion for the SGD_PI scheme
                new_delta = error_change

    # Weight decay is applied after estimating the delta and curvature, similar to
    # AdamW. See https://arxiv.org/abs/1711.05101 for details.
    if weight_decay != 0:
        pid_update.add_(param, alpha=weight_decay)

    alpha = lr if maximize else -lr
    param.add_(pid_update, alpha=alpha)

    if uses_error_buffer:
        # This branch handles the initializtion for the SGD_PI scheme (when called for
        # the first time)
        state["previous_error"] = error.clone().detach()
    if uses_delta_buffer and (new_delta is not None):
        state["previous_delta"] = new_delta.clone().detach()


def _sparse_pid_zero_init(
    param: torch.Tensor,
    state: dict,
    lr: float,
    weight_decay: float,
    Kp: float,
    Ki: float,
    Kd: float,
    ema_nu: float,
    init_type: PIDInitType,
    maximize: bool,
):
    """Analogous to _pid but with support for sparse gradients. This method implements
    updates under a buffer initialization of all zeros."""

    uses_error_buffer = Kp != 0 or Kd != 0
    uses_delta_buffer = Kd != 0

    error = param.grad
    assert error.is_sparse, "For dense updates, use _pid instead"

    error = error.coalesce()  # the update is non-linear so indices must be unique
    error_idx = error._indices()
    detached_error_values = error._values().clone().detach()

    if detached_error_values.numel() == 0:
        # Skip update for empty grad
        return

    if uses_error_buffer and ("previous_error" not in state):
        state["previous_error"] = torch.zeros_like(param)
    if uses_delta_buffer and ("previous_delta" not in state):
        state["previous_delta"] = torch.zeros_like(param)

    pid_update_values = torch.zeros_like(detached_error_values)

    if Ki != 0:
        pid_update_values.add_(detached_error_values, alpha=Ki)

    if uses_error_buffer:
        previous_error = state["previous_error"].sparse_mask(error)
        error_change_values = detached_error_values - previous_error._values()

        if Kp != 0:
            pid_update_values.add_(error_change_values, alpha=Kp)

        if Kd != 0:
            previous_delta = state["previous_delta"].sparse_mask(error)
            new_delta_values = previous_delta._values().mul(ema_nu).add(error_change_values, alpha=1 - ema_nu)
            delta_change = new_delta_values.sub(previous_delta._values())
            pid_update_values.add_(delta_change, alpha=Kd)

    pid_update = torch.sparse_coo_tensor(error_idx, pid_update_values, size=param.shape)

    # Weight decay is applied after estimating the delta and curvature, similar to
    # AdamW. See https://arxiv.org/abs/1711.05101 for details.
    if weight_decay != 0:
        observed_params = param.sparse_mask(error)
        pid_update.add_(observed_params, alpha=weight_decay)

    alpha = lr if maximize else -lr
    param.add_(pid_update, alpha=alpha)

    if "previous_error" in state:
        state["previous_error"][error_idx] = detached_error_values
    if "previous_delta" in state:
        state["previous_delta"][error_idx] = new_delta_values.clone().detach()


def _sparse_pid_sgd_pi_init(
    param: torch.Tensor,
    state: dict,
    lr: float,
    weight_decay: float,
    Kp: float,
    Ki: float,
    Kd: float,
    ema_nu: float,
    init_type: PIDInitType,
    maximize: bool,
):
    """Analogous to _pid but with support for sparse gradients. This method implements
    updates under a buffer initialization scheme that makes the first two updates of the
    algorithm (in each dimension) match those of a PI optimizer."""

    uses_error_buffer = Kp != 0 or Kd != 0
    uses_delta_buffer = Kd != 0

    error = param.grad
    assert error.is_sparse, "For dense updates, use _pid instead"

    error = error.coalesce()  # the update is non-linear so indices must be unique
    error_idx = error._indices()
    detached_error_values = error._values().clone().detach()

    if detached_error_values.numel() == 0:
        # Skip update for empty grad
        return

    if uses_error_buffer and ("previous_error" not in state):
        state["previous_error"] = torch.zeros_like(param)
        state["needs_error_initialization_mask"] = torch.ones_like(param, dtype=torch.bool)

    if uses_delta_buffer and ("previous_delta" not in state):
        state["previous_delta"] = torch.zeros_like(param)
        state["needs_delta_initialization_mask"] = torch.ones_like(param, dtype=torch.bool)

    pid_update_values = torch.zeros_like(detached_error_values)

    if Ki != 0:
        pid_update_values.add_(detached_error_values, alpha=Ki)

    if uses_error_buffer:

        previous_error_values = state["previous_error"].sparse_mask(error)._values()
        error_change_values = torch.where(
            state["needs_error_initialization_mask"].sparse_mask(error)._values(),
            torch.zeros_like(detached_error_values),
            detached_error_values - previous_error_values,
        )
        if Kp != 0:
            pid_update_values.add_(error_change_values, alpha=Kp)

        if Kd != 0:
            previous_delta_values = state["previous_delta"].sparse_mask(error)._values()
            new_delta_values = torch.where(
                state["needs_delta_initialization_mask"].sparse_mask(error)._values(),
                torch.zeros_like(detached_error_values),
                previous_delta_values.mul(ema_nu).add(error_change_values, alpha=1 - ema_nu),
            )
            pid_update_values.add_(new_delta_values.sub(previous_delta_values), alpha=Kd)

    pid_update = torch.sparse_coo_tensor(error_idx, pid_update_values, size=param.shape)

    # Weight decay is applied after estimating the delta and curvature, similar to
    # AdamW. See https://arxiv.org/abs/1711.05101 for details.
    if weight_decay != 0:
        observed_params = param.sparse_mask(error)
        pid_update.add_(observed_params, alpha=weight_decay)

    alpha = lr if maximize else -lr
    param.add_(pid_update, alpha=alpha)

    if uses_delta_buffer:
        # The update of the delta buffer works as follows:
        # - If delta has already been initialized, then we keep the EMA value stored in
        #   `new_delta_values`.
        # - If delta has not been initialized, then we check if the error buffer has
        #   been initialized.
        #   - If the error buffer has been initialized, we initialize delta with the
        #     first error change value `(e_1 - e_0)`.
        #   - Otherwise, we keep a value of zero in the delta buffer.
        # Finally, we mark the delta buffer as initialized for the .
        delta_buffer_update = torch.where(
            state["needs_delta_initialization_mask"].sparse_mask(error)._values(),
            torch.where(
                state["needs_error_initialization_mask"].sparse_mask(error)._values(),
                torch.zeros_like(detached_error_values),
                error_change_values,
            ),
            new_delta_values,
        )
        state["previous_delta"][error_idx] = delta_buffer_update.clone().detach()
        # The resulting value of the delta buffer initialization mask matches the
        # error buffer initialization mask at this stage:
        # - If the error buffer has been initialized (marked as False), then the delta
        #   buffer was either initialized previosuly or initialized in this step. Thus
        #   the delta buffer initialization mask can be marked as False.
        # - If the error buffer has not been initialized (marked as True), then the
        #   delta buffer is definitely not yet initialized, so retains the value True.
        # Note that error buffers for indices seen for the first time in this step are
        # only marked as initialized _after_ the update of the delta buffer. (See block
        # below.)
        state["needs_delta_initialization_mask"][error_idx] = state["needs_error_initialization_mask"][error_idx]
    if uses_error_buffer:
        state["previous_error"][error_idx] = detached_error_values
        state["needs_error_initialization_mask"][error_idx] *= False
