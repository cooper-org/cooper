"""The nuPI optimizer is a first-order optimization algorithm proposed in the ICML 2024
paper *On PI controllers for updating Lagrange multipliers in constrained optimization*
by Motahareh Sohrabi, Juan Ramirez, Tianyue H. Zhang, Simon Lacoste-Julien, and
Jose Gallego-Posada.
"""

import warnings
from collections.abc import Callable, Iterable
from enum import Enum
from typing import Optional

import torch


class nuPIInitType(Enum):
    r"""nuPI initialization types. This is used to determine how to initialize the
    error and derivative terms of the nuPI controller. The initialization scheme
    ``SGD`` ensures that the first step of ``nuPI(KP, KI)`` is equivalent to SGD with
    learning rate :math:`\eta \times K_I`. The ``ZEROS`` scheme yields a first step which
    corresponds to SGD with a learning rate of :math:`\eta \times (K_P + K_I)`.
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
        init_type: nuPIInitType = nuPIInitType.SGD,
        maximize: bool = False,
    ) -> None:
        r"""Implements the ``nuPI`` controller as a PyTorch optimizer.

        Controllers are designed to guide a system toward a desired state by adjusting a
        control variable. This is achieved by measuring the error, which is the
        difference between the desired and current states, and using this error to
        modify the control variable, thereby influencing the system.

        For this controller, the error signal is derived from the gradient of a loss
        function :math:`L` being optimized with respect to a parameter
        :math:`\vtheta`. Here, :math:`\vtheta` acts as the control variable, while the
        **gradient** of :math:`L` serves as the error signal, defined as
        :math:`\ve_t = \nabla L_t(\vtheta_t)`. The control objective of setting
        :math:`\nabla L_t(\vtheta_t) = 0` corresponds to finding a stationary point
        of the loss function, thereby minimizing (or maximizing) it.

        .. note::
            When applied to the Lagrange multipliers of a constrained minimization
            problem, the control state :math:`\nabla L_t(\vtheta_t)` corresponds to the
            gradient of the Lagrangian function with respect to the multipliers (e.g.,
            :math:`\nabla_{\vlambda} \Lag(\vx, \vlambda) = \vg(\vx)` for
            inequality-constrained problems). Setting this gradient to (less than or
            equal to) zero corresponds to finding a point that satisfies the
            constraints.

        The ``nuPI`` controller updates parameters as follows:

        .. math::
            \vxi_t &= \nu \vxi_{t-1} + (1 - \nu) \ve_t, \\
            \vtheta_1 &= \vtheta_0 - \eta (K_P \vxi_0 + K_I \ve_0), \\
            \vtheta_{t+1} &= \vtheta_t - \eta (K_I \ve_t + K_P (\vxi_t - \vxi_{t-1}))

        Here, :math:`\vxi_t` is a smoothed version of the error signal (:math:`\ve_t`),
        using an exponential moving average (EMA) with coefficient :math:`\nu`.
        :math:`K_P` and :math:`K_I` are the proportional and integral gains,
        respectively, while the learning rate :math:`\eta` is kept separate
        to allow comparison with other optimizers.

        Weight decay is applied based only on the error signal :math:`\ve_t`, following
        a similar approach to PyTorch's AdamW optimizer.

        When ``maximize=False``, the parameter update is multiplied by :math:`-1` before
        being applied.

        **Initialization Schemes**:
        The initialization of the ``nuPI`` controller requires specifying the initial
        smoothed error signal, :math:`\vxi_{-1}`, which impacts the first parameter
        update. Two initialization schemes are available:

        - ``nuPIInitType.ZEROS``: Initializes :math:`\vxi_{-1} = \vzero`. The first update rule becomes:

            .. math::
                \vtheta_1 = \vtheta_0 - \eta (K_P \ve_0 + K_I \ve_0) = \vtheta_0 - \eta (K_P + K_I) \ve_0.

        - ``nuPIInitType.SGD``: Initializes :math:`\vxi_{-1} = \ve_0`, producing a first step identical to SGD:

            .. math::
                \vxi_0 &= \ve_0, \\
                \vtheta_1 &= \vtheta_0 - \eta (K_P \ve_0 + K_I \ve_0) = \vtheta_0 - \eta K_I \ve_0.

        .. note::
            nuPI(:math:`\eta`, :math:`K_P=0`, :math:`K_I=1`, :math:`\nu=0`) corresponds
            to SGD with learning rate :math:`\eta`.

            nuPI(:math:`\eta`, :math:`K_P=1`, :math:`K_I=1`, :math:`\nu=0`) corresponds
            to the optimistic gradient method :cite:p:`popov1980modification`.

        Args:
            params: iterable of parameters to optimize, or dicts defining parameter groups.
            lr: learning rate.
            weight_decay: weight decay (L2 penalty). Defaults to 0.
            Kp: proportional gain. Defaults to 0.
            Ki: integral gain. Defaults to 1.
            ema_nu: EMA coefficient for the smoothed error signal. Defaults to 0,
                meaning no smoothing is applied.
            init_type: initialization scheme for :math:`\vxi_{-1}`. Defaults to
                ``nuPIInitType.SGD``, which matches the first step of SGD.
            maximize: whether to maximize the objective with respect to the parameters
                instead of minimizing. Defaults to ``False``.

        Raises:
            ValueError: If the learning rate, or weight decay is negative.
            ValueError: If the EMA coefficient is not in the range :math:`(-1, 1)`.
            ValueError: If the initialization type is invalid.
            NotImplementedError: If multiple parameter groups are used with non-scalar
                proportional and integral gains.

        Warnings:
            If a negative proportional or integral gain is used.
            If both proportional and integral gains are zero.
            If the EMA coefficient is negative.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not -1 < ema_nu < 1.0:
            raise ValueError(f"Invalid nu value: {ema_nu}")

        if init_type not in {nuPIInitType.ZEROS, nuPIInitType.SGD}:
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
    def disambiguate_update_function(is_grad_sparse: bool, init_type: nuPIInitType) -> Callable:
        if is_grad_sparse:
            if init_type == nuPIInitType.ZEROS:
                return _sparse_nupi_zero_init
            return _sparse_nupi_sgd_init
        if init_type == nuPIInitType.ZEROS:
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
