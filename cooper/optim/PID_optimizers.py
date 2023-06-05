"""Implementation of a PID controller as a PyTorch optimizer.
Parameters are control variables, and gradients are errors

Inspired by:
https://pytorch-optimizer.readthedocs.io/en/latest/_modules/torch_optimizer/pid.html

Intended to be used on the multipliers, as the P and D terms can reduce oscillations
common to min-max optimization problems.
"""

import torch
from torch.optim.optimizer import Optimizer


class PIDBase(Optimizer):
    r"""

    PID is x_t = x_0 + ...
    This implementation is recursive: x_t = x_{t-1} + ...

    The default for p=0, i=1, d=0 corresponds to SGD with learning rate lr
    If p!=0, i!=0, d=0, then the optimizer corresponds to an optimistic gradient descent

    Mention memory  and computational complexity wrt SGD

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
        ...
    """

    def __init__(
        self,
        params,
        lr: float,
        weight_decay: float = 0.0,
        proportional: float = 0.0,
        integral: float = 1.0,
        derivative: float = 0.0,
        maximize: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        if proportional < 0.0:
            raise ValueError("Invalid PID proportional value: {}".format(proportional))
        if integral < 0.0:
            raise ValueError("Invalid PID integral value: {}".format(integral))
        if derivative < 0.0:
            raise ValueError("Invalid PID derivative value: {}".format(derivative))
        if all([proportional == 0.0, integral == 0.0, derivative == 0.0]):
            raise ValueError("Invalid PID parameters: all are zero")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            proportional=proportional,
            integral=integral,
            derivative=derivative,
            maximize=maximize,
        )

        super().__init__(params, defaults)


class PID(PIDBase):
    # TODO(juan43ramirez): ema_beta

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
            weight_decay = group["weight_decay"]
            proportional = group["proportional"]
            integral = group["integral"]
            derivative = group["derivative"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("PID optimizer does not support sparse gradients. Consider SparsePID instead.")

                grad = p.grad
                state = self.state[p]

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

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
                    previous_direction = state["previous_direction"]
                    change = grad.sub(previous_direction)
                    curvature = 0
                else:
                    change = grad.sub(state["previous_direction"])
                    curvature = change.sub(state["previous_change"])

                d_p = grad.mul(integral)

                if proportional != 0:
                    d_p.add_(change, alpha=proportional)
                if derivative != 0:
                    d_p.add_(curvature, alpha=derivative)

                if maximize:
                    d_p.mul_(-1)

                p.add_(d_p, alpha=-group["lr"])

                if len(state) == 0:
                    # Only the I term was used for the first update. For the next step,
                    # the current update direction will be used to compute the P term.
                    # We do not initialize `previous_change` as a convention to indicate
                    # that the D term should not be used in the following update.
                    state["previous_direction"] = grad.clone().detach()
                else:
                    state["previous_direction"] = grad.clone().detach()
                    state["previous_change"] = change.clone().detach()

        return loss


class SparsePID(PIDBase):
    r"""
    Supports sparse gradient updates. Inspired by SparseAdam from PyTorch.
    https://github.com/pytorch/pytorch/blob/0bb2b015414214e8874d4c31188eb2fd883da402/torch/optim/_functional.py#L22
    """

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
            weight_decay = group["weight_decay"]
            proportional = group["proportional"]
            integral = group["integral"]
            derivative = group["derivative"]
            maximize = group["maximize"]

        for p in group["params"]:
            if p.grad is None:
                continue
            if not p.grad.is_sparse:
                raise RuntimeError("SparsePID optimizer only supports sparse gradients. Consider PID instead.")

            grad = p.grad
            state = self.state[p]

            grad = grad.coalesce()  # the update is non-linear so indices must be unique
            grad_indices = grad._indices()
            grad_values = grad._values()
            if grad_values.numel() == 0:
                # Skip update for empty grad
                continue
            size = grad.size()

            def make_sparse(values):
                constructor = grad.new
                if grad_indices.dim() == 0 or values.dim() == 0:
                    return constructor().resize_as_(grad)
                return constructor(grad_indices, values, size)

            if weight_decay != 0:
                grad_values.add_(make_sparse(p.data), alpha=weight_decay)

            if len(state) == 0:
                # At this stage, there is not enough information to compute the
                # change in direction for the P term nor the curvature for the D
                # term. Therefore, only the I term is used for the first update.
                change_values = 0
                curvature_values = 0
            elif "previous_change" not in state:
                assert "previous_direction" in state
                # Using the previous update direction to compute the P term, but
                # there is not enough information to compute the D term.
                previous_direction = state["previous_direction"].sparse_mask(grad)._values()
                change_values = grad_values.sub(previous_direction)
                curvature_values = 0
            else:
                previous_direction = state["previous_direction"].sparse_mask(grad)._values()
                change_values = grad.sub(previous_direction)

                previous_change = state["previous_change"].sparse_mask(grad)._values()
                curvature_values = change_values.sub(previous_change)

            curvature_values = change_values.sub(previous_change)

            d_p_values = grad_values.mul(integral)

            if proportional != 0:
                d_p_values.add_(change_values, alpha=proportional)
            if derivative != 0:
                d_p_values.add_(curvature_values, alpha=derivative)

            if maximize:
                d_p_values.mul_(-1)

            p.add_(make_sparse(d_p_values), alpha=-group["lr"])

            # TODO(juan43ramirez): To be accurate, each parameter should have its own
            # notion of whether it has been updated or not

            if len(state) == 0:
                # We do not initialize `previous_change` as a convention to indicate
                # that the D term should not be used in the following update.

                # Only the state associated with observed gradients is updated.
                # TODO(juan43ramirez): is this an efficient way to do this?
                state["previous_direction"][grad_indices] = grad_values.clone().detach()
            else:
                # Only the state associated with observed gradients is updated.
                # TODO(juan43ramirez): is this an efficient way to do this?
                state["previous_direction"][grad_indices] = grad_values.clone().detach()
                state["previous_change"][grad_indices] = change_values.clone().detach()

        return loss
