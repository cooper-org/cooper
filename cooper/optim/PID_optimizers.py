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
    # TODO
    pass


class PID(Optimizer):
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
    # TODO(juan43ramirez): ema_beta

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

        for p in params:
            if p.grad.is_sparse:
                raise RuntimeError("PID optimizer does not support sparse gradients. Consider SparsePID instead.")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            proportional=proportional,
            integral=integral,
            derivative=derivative,
            maximize=maximize,
        )

        super().__init__(params, defaults)

    # @_use_grad_for_differentiable
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

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Lazy state initialization
                if len(state) == 0:
                    # Previous update direction.
                    # TODO(juan43ramirez): for EMAs, initializing to zero makes the
                    # estimate of the EMA biased. This would require a correction term
                    # like with Adam or initializing to the first gradient.
                    state["previous_direction"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    # Previous update difference. That is the difference between the
                    # previous update and the update before that. Initialized to zero.
                    state["previous_direction_difference"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                previous_direction = state["previous_direction"]
                change = grad.sub(previous_direction)

                previous_change = state["previous_change"]
                curvature = change.sub(previous_change)

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                d_p = grad.mul(integral)

                if proportional != 0:
                    d_p.add_(change, alpha=proportional)
                if derivative != 0:
                    d_p.add_(curvature, alpha=derivative)

                if group["maximize"]:
                    d_p.mul_(-1)

                p.add_(d_p, alpha=-group["lr"])

                # Update state
                state["previous_direction"] = grad.clone().detach()
                state["previous_change"] = change.clone().detach()

        return loss


class SparsePID(Optimizer):
    r"""

    Shorter description of SparsePID.

    Supports sparse gradient updates. Inspired by SparseAdam from PyTorch.
    https://github.com/pytorch/pytorch/blob/0bb2b015414214e8874d4c31188eb2fd883da402/torch/optim/_functional.py#L22

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
        ...
    """
    # TODO(juan43ramirez): ema_beta

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

        for p in params:
            if not p.grad.is_sparse:
                raise RuntimeError("SparsePID optimizer does not support dense gradients. Consider PID instead.")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            proportional=proportional,
            integral=integral,
            derivative=derivative,
            maximize=maximize,
        )

        super().__init__(params, defaults)

    # @_use_grad_for_differentiable
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

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]

            # Lazy state initialization
            if len(state) == 0:
                # Previous update direction.
                # TODO(juan43ramirez): for EMAs, initializing to zero makes the
                # estimate of the EMA biased. This would require a correction term
                # like with Adam or initializing to the first gradient.
                state["previous_direction"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Previous update difference. That is the difference between the
                # previous update and the update before that. Initialized to zero.
                state["previous_direction_difference"] = torch.zeros_like(p, memory_format=torch.preserve_format)

            #  grad = grad if not maximize else -grad

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

            previous_direction = state["previous_direction"].sparse_mask(grad)._values()
            change_values = grad_values.sub(previous_direction)

            previous_change = state["previous_change"].sparse_mask(grad)._values()
            curvature_values = change_values.sub(previous_change)

            if weight_decay != 0:
                grad_values.add_(make_sparse(p.data), alpha=weight_decay)

            d_p_values = grad_values.mul(integral)

            if proportional != 0:
                d_p_values.add_(change_values, alpha=proportional)
            if derivative != 0:
                d_p_values.add_(curvature_values, alpha=derivative)

            if group["maximize"]:
                d_p_values.mul_(-1)

            p.add_(make_sparse(d_p_values), alpha=-group["lr"])

            # Update state. We only modify the parts of the state associated with
            # observed gradients.
            with torch.no_grad():
                state["previous_direction"].sparse_mask(grad)._values().copy_(grad_values)
                state["previous_change"].sparse_mask(grad)._values().copy_(change_values)

        return loss
