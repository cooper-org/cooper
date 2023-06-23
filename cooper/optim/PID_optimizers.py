"""Implementation of a PID controller as a PyTorch optimizer.
Parameters are control variables, and gradients are errors

Inspired by:
https://pytorch-optimizer.readthedocs.io/en/latest/_modules/torch_optimizer/pid.html

Intended to be used on the multipliers, as the P and D terms can reduce oscillations
common to min-max optimization problems.
"""

import warnings

import torch

# TODO(juan43ramirez): implement a PID optimizer which incorporates some filtering
# This could be by using EMAs of the error terms


class PID(torch.optim.Optimizer):
    r"""
    TODO(juan43ramirez): complete docstring

    PID is x_t = x_0 + ...
    This implementation is recursive: x_t = x_{t-1} + ...

    The default for p=0, i=1, d=0 corresponds to SGD with learning rate lr
    If p!=0, i!=0, d=0, then the optimizer corresponds to an optimistic gradient descent

    Mention memory  and computational complexity wrt SGD

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
        ...
    """

    # TODO: Kp -> Kp

    def __init__(
        self,
        params,
        lr: float,
        weight_decay: float = 0.0,
        Kp: float = 0.0,
        Ki: float = 1.0,
        Kd: float = 0.0,
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

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
            maximize=maximize,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
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
            Kp = group["Kp"]
            Ki = group["Ki"]
            Kd = group["Kd"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                func = _sparse_pid if p.grad.is_sparse else _pid
                func(p, state, lr, weight_decay, Kp, Ki, Kd, maximize)

        return loss


def _estimate_change_and_curvature(grad, state):
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


def _pid(param, state, lr, weight_decay, Kp, Ki, Kd, maximize):
    if param.grad.is_sparse:
        raise RuntimeError("PID optimizer does not support sparse gradients. Consider SparsePID instead.")

    grad = param.grad

    change, curvature = _estimate_change_and_curvature(grad, state)

    d_p = grad.mul(Ki)

    if Kp != 0:
        d_p.add_(change, alpha=Kp)
    if Kd != 0:
        d_p.add_(curvature, alpha=Kd)

    # Weight decay is applied after estimating the change and curvsture, similar to
    # AdamW. See https://arxiv.org/abs/1711.05101 for details.
    if weight_decay != 0:
        d_p.add_(param, alpha=weight_decay)

    if maximize:
        d_p.mul_(-1)

    param.add_(d_p, alpha=-lr)

    if len(state) == 0:
        # Only the I term was used for the first update. For the next step,
        # the current update direction will be used to compute the P term.
        # We do not initialize `previous_change` as a convention to indicate
        # that the D term should not be used in the following update.
        state["previous_direction"] = grad.clone().detach()
    else:
        state["previous_direction"] = grad.clone().detach()
        state["previous_change"] = change.clone().detach()


def _sparse_pid(params, state, lr, weight_decay, Kp, Ki, Kd, maximize):
    r"""
    Supports sparse gradient updates. Inspired by SparseAdam from PyTorch.
    https://github.com/pytorch/pytorch/blob/release/2.0/torch/optim/_functional.py
    """
    grad = params.grad

    grad = grad.coalesce()  # the update is non-linear so indices must be unique
    grad_indices = grad._indices()
    grad_values = grad._values()
    if grad_values.numel() == 0:
        # Skip update for empty grad
        return
    size = grad.size()

    def make_sparse(values):
        constructor = grad.new
        if grad_indices.dim() == 0 or values.dim() == 0:
            return constructor().resize_as_(grad)
        return constructor(grad_indices, values, size)

    if len(state) == 0:
        # NOTE: considering a *dense* state. Note that IndexedMultipliers are
        # stored in a dense representation as well.
        state["steps"] = torch.zeros_like(params, dtype=torch.int)
        state["previous_direction"] = torch.zeros_like(params)
        state["previous_change"] = torch.zeros_like(params)

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
        p_values = torch.index_select(params, 0, grad_indices[0])
        d_p_values.add_(p_values, alpha=weight_decay)

    if maximize:
        d_p_values.mul_(-1)

    params.add_(make_sparse(d_p_values), alpha=-lr)

    # Update the step counter for observed parameters.
    state["steps"].add_(make_sparse(torch.ones_like(grad_values, dtype=torch.int)))

    # Update the previous direction and change for observed parameters. We
    # always store `previous_direction` for the next update. `previous_change`
    # is only used for the second update, so we store it using ``
    state["previous_direction"][grad_indices] = grad_values.clone().detach()
    state["previous_change"][grad_indices] = change_values.mul(is_after_first_update.float()).clone().detach()
