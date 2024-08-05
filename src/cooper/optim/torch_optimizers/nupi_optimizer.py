r"""The nuPI optimizer is a first-order optimization algorithm proposed in the paper
"On PI controllers for updating Lagrange multipliers in constrained optimization." by
Motahareh Sohrabi, Juan Ramirez, Tianyue H. Zhang, Simon Lacoste-Julien, and
Jose Gallego-Posada.

nuPI generalizes various popular first-order optimization algorithms, including gradient
descent, gradient descent with Polyak and Nesterov momentum, the optimistic gradient
method, and PI controllers. It's benefits when updating Lagrange multipliers in
Lagrangian constrained optimization are discussed in the paper.

For a detailed explanation of the $\nu$PI algorithm, see the paper:
*On PI Controllers for Updating Lagrange Multipliers in Constrained Optimization* at
`ICML 2024 <https://openreview.net/forum?id=1khG2xf1yt>`_.
"""

import warnings
from collections.abc import Iterable
from enum import Enum
from typing import Callable, Optional

import torch


class InitType(Enum):
    r"""nuPI initialization types. This is used to determine how to initialize the
    error and derivative terms of the nuPI controller. The initialization scheme
    `SGD` ensures that the first step of `nuPI(KP, KI)` is equivalent to SGD with
    learning rate :math:`\text{lr} K_I`.
    """

    ZEROS = 0
    SGD = 1


class nuPI(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        weight_decay: Optional[float] = 0.0,
        Kp: Optional[torch.Tensor] = 0.0,
        Ki: Optional[torch.Tensor] = 1.0,
        ema_nu: float = 0.0,
        init_type: InitType = InitType.SGD,
        maximize: bool = False,
    ) -> None:
        r"""Implements a nuPI controller as a PyTorch optimizer.

        The error signal used for the nuPI controller is the gradient of a cost function
        :math:`L` being optimized, with parameter :math:`\theta`. We treat :math:`\theta`
        as the control variable, and the gradient of :math:`L` as the error signal. The
        error signal at time :math:`t` is :math:`e_t = \nabla L_t(\theta_t)`. Note that
        the function :math:`L_t` may change over time.

        When ``maximize=False``, the parameter update is multiplied by :math:`-1` before
        being applied.

        The execution of the nuPI controller is given by:

        .. math::
            \xi_t &= \nu \xi_{t-1} + (1 - \nu) e_t \\
            \theta_1 &= \theta_0 - \text{lr} (K_P \xi_0 + K_I e_0) \\
            \theta_{t+1} &= \theta_t - \text{lr} (K_I e_t + K_P (\xi_t - \xi_{t-1})),

        where :math:`K_P`, :math:`K_I` are the proportional and integral gains,
        respectively. We keep the learning rate :math:`\text{lr}` as a separate
        parameter to facilitate comparison with other optimizers.

        .. note::
            The optimizer state is initialized with :math:`\xi_{-1} = 0`.

        .. note::
            Setting :math:`K_P=0`, :math:`K_I=1` and :math:`\nu=0` corresponds to SGD
            with learning rate :math:`\text{lr}`.

            Setting :math:`K_P=1`, :math:`K_I=1` and :math:`\nu=0` corresponds to the
            optimistic gradient method.

        Args:
            params: iterable of parameters to optimize or dicts defining parameter groups
            lr: learning rate
            weight_decay: weight decay (L2 penalty)
            Kp: proportional gain
            Ki: integral gain
            ema_nu: EMA coefficient
            init_type: initialization scheme
            maximize: whether to maximize or minimize the loss
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not -1 < ema_nu < 1.0:
            raise ValueError(f"Invalid nu value: {ema_nu}")

        if init_type not in {InitType.ZEROS, InitType.SGD}:
            raise ValueError(f"Invalid init_type: {init_type}")

        if not isinstance(Kp, torch.Tensor):
            Kp = torch.tensor(Kp)
        if not isinstance(Ki, torch.Tensor):
            Ki = torch.tensor(Ki)

        if torch.any(Kp < 0.0):
            warnings.warn(f"Using a negative Kp coefficient: {Kp}")
        if torch.any(Ki < 0.0):
            warnings.warn(f"Using a negative Ki coefficient: {Kp}")
        if torch.all(Kp == 0.0) and torch.all(Ki == 0.0):
            warnings.warn("All PI coefficients are zero")
        if ema_nu < 0:
            warnings.warn("nuPI optimizer instantiated with negative EMA coefficient")

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "Kp": Kp,
            "Ki": Ki,
            "ema_nu": ema_nu,
            "maximize": maximize,
            "init_type": init_type,
        }

        super().__init__(params, defaults)

        if len(self.param_groups) > 1 and Kp.shape != torch.Size([1]):
            raise NotImplementedError("When using multiple parameter groups, Kp and Ki must be scalars")

    @staticmethod
    def disambiguate_update_function(is_grad_sparse: bool, init_type: InitType) -> Callable:
        if is_grad_sparse:
            if init_type == InitType.ZEROS:
                return _sparse_nupi_zero_init
            return _sparse_nupi_sgd_init
        if init_type == InitType.ZEROS:
            return _nupi_zero_init
        return _nupi_sgd_init

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
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

                update_function = self.disambiguate_update_function(p.grad.is_sparse, group["init_type"])
                update_function(
                    param=p,
                    state=self.state[p],
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    Kp=group["Kp"],
                    Ki=group["Ki"],
                    ema_nu=group["ema_nu"],
                    maximize=group["maximize"],
                )

        return loss

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "needs_error_initialization_mask" in state:
                    # Need to convert to bool explicitly since torch might have loaded
                    # it as a float tensor and the `torch.where` calls would fail
                    state["needs_error_initialization_mask"] = state["needs_error_initialization_mask"].bool()


def _nupi_zero_init(
    param: torch.Tensor,
    state: dict,
    lr: float,
    weight_decay: float,
    Kp: torch.Tensor,
    Ki: torch.Tensor,
    ema_nu: float,
    maximize: bool,
) -> None:
    """Applies a nuPI step update to `param`."""
    error = param.grad
    detached_error = error.clone().detach()
    assert not error.is_sparse, "For sparse updates, use _sparse_nupi instead"

    xit_m1_coef = Kp * (1 - ema_nu)
    if "xi" not in state and xit_m1_coef.ne(0).any():
        state["xi"] = torch.zeros_like(param)

    nupi_update = torch.zeros_like(param)

    et_coef = Ki + Kp * (1 - ema_nu)
    if et_coef.ne(0).any():
        nupi_update.add_(detached_error.mul(et_coef))

    if xit_m1_coef.ne(0).any():
        nupi_update.sub_(state["xi"].mul(xit_m1_coef))

    # Weight decay is applied after estimating the error change, similar to AdamW.
    # See https://arxiv.org/abs/1711.05101 for details.
    if weight_decay != 0:
        nupi_update.add_(param, alpha=-weight_decay if maximize else weight_decay)

    alpha = lr if maximize else -lr
    param.add_(nupi_update, alpha=alpha)

    if "xi" in state and xit_m1_coef.ne(0).any():
        state["xi"].mul_(ema_nu).add_(detached_error, alpha=1 - ema_nu)


def _sparse_nupi_zero_init(
    param: torch.Tensor,
    state: dict,
    lr: float,
    weight_decay: float,
    Kp: float,
    Ki: float,
    ema_nu: float,
    maximize: bool,
) -> None:
    """Analogous to _nupi but with support for sparse gradients. This function implements
    updates based on a zero initialization scheme.
    """
    error = param.grad
    assert error.is_sparse, "For dense updates, use _nupi instead"

    error = error.coalesce()  # the update is non-linear so indices must be unique
    error_indices = error.indices()
    detached_error_values = error._values().clone().detach()

    if detached_error_values.numel() == 0:
        # Skip update for empty grad
        return

    Ki_values = Ki[error_indices] if Ki.numel() > 1 else Ki
    Kp_values = Kp[error_indices] if Kp.numel() > 1 else Kp

    nupi_update_values = torch.zeros_like(detached_error_values)

    et_coef = Ki_values + Kp_values * (1 - ema_nu)
    xit_m1_coef = Kp_values * (1 - ema_nu)

    if "xi" not in state and (xit_m1_coef).ne(0).any():
        state["xi"] = torch.zeros_like(param)

    et_coef_values = et_coef[error_indices] if et_coef.numel() > 1 else et_coef
    xit_m1_coef_values = xit_m1_coef[error_indices] if xit_m1_coef.numel() > 1 else xit_m1_coef

    if et_coef_values.ne(0).any():
        nupi_update_values.add_(detached_error_values.mul(et_coef_values))

    if xit_m1_coef_values.ne(0).any():
        xi_values = state["xi"].sparse_mask(error)._values()
        nupi_update_values.sub_(xi_values.mul(xit_m1_coef))

    nupi_update = torch.sparse_coo_tensor(error_indices, nupi_update_values, size=param.shape)

    # Weight decay is applied after estimating the delta and curvature, similar to
    # AdamW. See https://arxiv.org/abs/1711.05101 for details.
    if weight_decay != 0:
        observed_params = param.sparse_mask(error)
        nupi_update.add_(observed_params, alpha=-weight_decay if maximize else weight_decay)

    alpha = lr if maximize else -lr
    param.add_(nupi_update, alpha=alpha)

    if "xi" in state and xit_m1_coef_values.ne(0).any():
        state["xi"][error_indices] = xi_values.mul(ema_nu).add(detached_error_values, alpha=1 - ema_nu)


def _nupi_sgd_init(
    param: torch.Tensor,
    state: dict,
    lr: float,
    weight_decay: float,
    Kp: torch.Tensor,
    Ki: torch.Tensor,
    ema_nu: float,
    maximize: bool,
) -> None:
    """Applies a nuPI step update to `param`."""
    error = param.grad
    detached_error = error.clone().detach()
    assert not error.is_sparse, "For sparse updates, use _sparse_nupi_* instead"

    uses_ki_term = Ki.ne(0).any()
    uses_kp_term = (Kp * (1 - ema_nu)).ne(0).any()

    nupi_update = torch.zeros_like(param)

    if uses_ki_term:
        nupi_update.add_(error.mul(Ki))

    if uses_kp_term:
        if "xi" in state:
            kp_term_contribution = (1 - ema_nu) * (detached_error - state["xi"])
            nupi_update.add_(kp_term_contribution.mul(Kp))
        else:
            # First step is designed to match GD, so no need to add contribution here
            pass

    # Weight decay is applied after estimating the error change, similar to AdamW.
    # See https://arxiv.org/abs/1711.05101 for details.
    if weight_decay != 0:
        nupi_update.add_(param, alpha=-weight_decay if maximize else weight_decay)

    alpha = lr if maximize else -lr
    param.add_(nupi_update, alpha=alpha)

    if uses_kp_term:
        if "xi" not in state:
            # Initialize xi_0 = 0
            state["xi"] = torch.zeros_like(param)
        else:
            state["xi"].mul_(ema_nu).add_(detached_error, alpha=1 - ema_nu)


def _sparse_nupi_sgd_init(
    param: torch.Tensor,
    state: dict,
    lr: float,
    weight_decay: float,
    Kp: float,
    Ki: float,
    ema_nu: float,
    maximize: bool,
) -> None:
    """Analogous to _nupi but with support for sparse gradients. This function implements
    updates based on a "SGD" initialization scheme that makes the first step of nuPI
    (on each coordinate) match that of SGD.
    """
    error = param.grad
    assert error.is_sparse, "For dense updates, use _nupi instead"

    error = error.coalesce()  # the update is non-linear so indices must be unique
    error_indices = error.indices()
    detached_error_values = error._values().clone().detach()

    if detached_error_values.numel() == 0:
        # Skip update for empty grad
        return

    filtered_Ki_values = Ki[error_indices] if Ki.numel() > 1 else Ki
    filtered_Kp_values = Kp[error_indices] if Kp.numel() > 1 else Kp

    uses_ki_term = filtered_Ki_values.ne(0).any()
    uses_kp_term = (filtered_Kp_values * (1 - ema_nu)).ne(0).any()

    nupi_update_values = torch.zeros_like(detached_error_values)

    if "xi" not in state and uses_kp_term:
        state["xi"] = torch.zeros_like(param)
        state["needs_error_initialization_mask"] = torch.ones_like(param, dtype=torch.bool)

    if uses_ki_term:
        nupi_update_values.add_(detached_error_values.mul(filtered_Ki_values))

    if uses_kp_term:
        previous_xi_values = state["xi"].sparse_mask(error)._values()
        proportional_term_contribution = torch.where(
            state["needs_error_initialization_mask"].sparse_mask(error)._values(),
            torch.zeros_like(detached_error_values),  # If state has not been initialized, xi_0 = 0
            (1 - ema_nu) * (detached_error_values - previous_xi_values),  # Else, we use recursive update
        )
        nupi_update_values.add_(proportional_term_contribution.mul(filtered_Kp_values))

    nupi_update = torch.sparse_coo_tensor(error_indices, nupi_update_values, size=param.shape)

    # Weight decay is applied after estimating the delta and curvature, similar to
    # AdamW. See https://arxiv.org/abs/1711.05101 for details.
    if weight_decay != 0:
        observed_params = param.sparse_mask(error)
        nupi_update.add_(observed_params, alpha=-weight_decay if maximize else weight_decay)

    alpha = lr if maximize else -lr
    param.add_(nupi_update, alpha=alpha)

    if "xi" in state and uses_kp_term:
        state["xi"][error_indices] = previous_xi_values.mul(ema_nu).add(detached_error_values, alpha=1 - ema_nu)
        state["needs_error_initialization_mask"][error_indices] *= False
