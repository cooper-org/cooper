"""Classes for modeling dual variables (e.g. Lagrange multipliers)."""
import abc
from typing import Optional

import torch


class ConstantMultiplier:
    """
    Constant (non-trainable) multiplier class used for penalized formulations.

    Args:
        init: Value of the multiplier.
    """

    def __init__(self, init: torch.Tensor):
        if init.requires_grad:
            raise ValueError("Constant multiplier should not be trainable.")
        self.weight = init
        self.device = init.device

    def __call__(self):
        """Return the current value of the multiplier."""
        return self.weight

    def parameters(self):
        """Return an empty iterator for consistency with multipliers which are
        :py:class:`~torch.nn.Module`."""
        return iter(())

    def state_dict(self):
        return {"weight": self.weight}

    def load_state_dict(self, state_dict):
        self.weight = state_dict["weight"]


class ExplicitMultiplier(torch.nn.Module):
    """
    An explicit multiplier holds a :py:class:`~torch.nn.parameter.Parameter` which
    contains (explicitly) the value of the Lagrange multipliers associated with a
    :py:class:`~cooper.constraints.ConstraintGroup` in a
    :py:class:`~cooper.cmp.ConstrainedMinimizationProblem`.

    .. warning::
        When `restart_on_feasible=True`, the entries of the multiplier which correspond
        to feasible constraints in the :py:class:`~cooper.constraints.ConstraintGroup`
        are reset to a default value (typically zero) by the
        :py:meth:`~cooper.multipliers.ExplicitMultiplier.post_step_` method. Note that
        we do **not** perform any modification to the dual optimizer associated with
        this multiplier.
        We discourage the use of `restart_on_feasible` along with stateful optimizers
        (such as :py:class:`~torch.optim.SGD` with momentum or
        :py:class:`~torch.optim.Adam`) since this combination can lead to the optimizer
        buffers becoming stale/wrong for the entries of the multiplier which have been
        reset due to the feasibility of their associated constraint.

    Args:
        init: Initial value of the multiplier.
        enforce_positive: Whether to enforce non-negativity on the values of the
            multiplier.
        restart_on_feasible: Whether to restart the value of the multiplier (to 0 by
            default) when the constrain is feasible. This is only supported for
            inequality constraints (i.e. enforce_positive=True)
    """

    def __init__(
        self,
        init: torch.Tensor,
        *,
        enforce_positive: bool = False,
        restart_on_feasible: bool = False,
        default_restart_value: float = 0.0,
    ):
        super().__init__()

        self.enforce_positive = enforce_positive
        self.restart_on_feasible = restart_on_feasible

        if self.enforce_positive and any(init < 0):
            raise ValueError("For inequality constraint, all entries in multiplier must be non-negative.")

        if not self.enforce_positive and restart_on_feasible:
            raise ValueError("Restart on feasible is not supported for equality constraints.")

        if (default_restart_value < 0) and restart_on_feasible:
            raise ValueError("Default restart value must be positive.")

        if (default_restart_value > 0) and not restart_on_feasible:
            raise ValueError("Default restart was provided but `restart_on_feasible=False`.")

        self.weight = torch.nn.Parameter(init)
        self.device = self.weight.device
        self.default_restart_value = default_restart_value

    @property
    def implicit_constraint_type(self):
        return "ineq" if self.enforce_positive else "eq"

    def post_step_(self, feasible_indices: Optional[torch.Tensor] = None):
        """
        Post-step function for multipliers. This function is called after each step of
        the dual optimizer, and ensures that (if required) the multipliers are
        non-negative. It also restarts the value of the multipliers for constraints that
        are feasible.

        Args:
            feasible_indices: Indices or binary masks denoting the feasible constraints.
        """

        if self.enforce_positive:
            # Ensures non-negativity for multipliers associated with inequality constraints.
            self.weight.data = torch.relu(self.weight.data)

            # TODO(juan43ramirez): Document https://github.com/cooper-org/cooper/issues/28
            # about the pitfalls of using dual_restars with stateful optimizers.
            if self.restart_on_feasible and feasible_indices is not None:
                self.weight.data[feasible_indices, ...] = self.default_restart_value
                if self.weight.grad is not None:
                    self.weight.grad[feasible_indices, ...] = 0.0

    def state_dict(self):
        _state_dict = super().state_dict()
        _state_dict["enforce_positive"] = self.enforce_positive
        _state_dict["restart_on_feasible"] = self.restart_on_feasible
        _state_dict["default_restart_value"] = self.default_restart_value
        return _state_dict

    def load_state_dict(self, state_dict):
        self.enforce_positive = state_dict.pop("enforce_positive")
        self.restart_on_feasible = state_dict.pop("restart_on_feasible")
        self.default_restart_value = state_dict.pop("default_restart_value") 
        super().load_state_dict(state_dict)
        self.device = self.weight.device


class DenseMultiplier(ExplicitMultiplier):
    """
    This is the simplest kind of trainable Lagrange multiplier.
    :py:class:`~cooper.multipliers.DenseMultiplier`\\s are suitable for low to mid-scale
    :py:class:`~cooper.constraints.ConstraintGroup`\\s for which all the constraints
    in the group are measured constantly.

    For large-scale :py:class:`~cooper.constraints.ConstraintGroup`\\s (for example,
    one constraint per training example) you may consider using a
    :py:class:`~cooper.multipliers.SparseMultiplier`.
    """

    def forward(self):
        """Return the current value of the multiplier."""
        return self.weight

    def __repr__(self):
        return f"DenseMultiplier({self.implicit_constraint_type}, shape={self.weight.shape})"


class SparseMultiplier(ExplicitMultiplier):
    """
    Sparse multipliers extend the functionality of
    :py:class:`~cooper.multipliers.DenseMultiplier`\\s to cases where the number of
    constraints in the :py:class:`~cooper.constraints.ConstraintGroup` is too large.
    This situation may arise, for example, when imposing point-wise constraints over all
    the training samples in a learning task.

    In such cases, it might be computationally prohibitive to measure the value for all
    the constraints in the :py:class:`~cooper.constraints.ConstraintGroup` and one may
    typically resort to sampling. :py:class:`~cooper.multipliers.SparseMultiplier`\\s
    enable time-efficient retrieval of the multipliers for the sampled constraints only,
    and memory-efficient sparse gradients.
    """

    def forward(self, indices: torch.Tensor):
        """Return the current value of the multiplier at the provided indices."""

        if indices.dtype != torch.long:
            # Not allowing for boolean "indices", which are treated as indices by
            # torch.nn.functional.embedding and *not* as masks.
            raise ValueError("Indices must be of type torch.long.")

        return torch.nn.functional.embedding(indices, self.weight, sparse=True)

    def __repr__(self):
        return f"SparseMultiplier({self.implicit_constraint_type}, shape={self.weight.shape})"


class ImplicitMultiplier(torch.nn.Module, metaclass=abc.ABCMeta):
    """
    An implicit multiplier is a :py:class:`~torch.nn.Module` that computes the value
    of a Lagrange multiplier associated with a
    :py:class:`~cooper.constraints.ConstraintGroup` based on "features" for each
    constraint. The multiplier is _implicitly_  represented by the features of its
    associated constraint as well as the computation that takes place in the
    :py:meth:`~cooper.multipliers.ImplicitMultiplier.forward` method.

    Thanks to their functional nature, implicit multipliers can allow for
    (approximately) representing _infinitely_ many constraints. This feature is based on
     the Lagrange "multiplier model" proposed by :cite:p:`narasimhan2020multiplier`.
    """

    @abc.abstractmethod
    def forward(self):
        pass

    def post_step_(self):
        """
        Post-step function for multipliers. This function is called after each step of
        the dual optimizer, and allows for additional post-processing of the implicit
        multiplier module or its parameters.
        """
        pass
