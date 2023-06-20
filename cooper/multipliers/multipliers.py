"""Classes for modeling dual variables (e.g. Lagrange multipliers)."""
import abc
import warnings
from typing import Optional

import torch

from cooper.constraints.constraint_state import ConstraintType


class Multiplier(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self):
        """Return the current value of the multiplier."""
        pass

    @abc.abstractmethod
    def post_step_(self):
        """
        Post-step function for multipliers. This function is called after each step of
        the dual optimizer, and allows for additional post-processing of the implicit
        multiplier module or its parameters.
        """
        pass


class ExplicitMultiplier(Multiplier):
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

        self.weight = torch.nn.Parameter(init)
        self.device = self.weight.device
        self.default_restart_value = default_restart_value

        self.strictly_feasible_indices = None

        self.base_sanity_checks()

    def base_sanity_checks(self):
        if self.enforce_positive and torch.any(self.weight.data < 0):
            raise ValueError("For inequality constraint, all entries in multiplier must be non-negative.")

        if not self.enforce_positive and self.restart_on_feasible:
            raise ValueError("Restart on feasible is not supported for equality constraints.")

        if (self.default_restart_value < 0) and self.restart_on_feasible:
            raise ValueError("Default restart value must be positive.")

        if (self.default_restart_value > 0) and not self.restart_on_feasible:
            raise ValueError("Default restart was provided but `restart_on_feasible=False`.")

    @property
    def implicit_constraint_type(self):
        return ConstraintType.INEQUALITY if self.enforce_positive else ConstraintType.EQUALITY

    def post_step_(self):
        """
        Post-step function for multipliers. This function is called after each step of
        the dual optimizer, and ensures that (if required) the multipliers are
        non-negative. It also restarts the value of the multipliers for inequality
        constraints that are strictly feasible.
        """

        if self.enforce_positive:
            # Ensures non-negativity for multipliers associated with inequality constraints.
            self.weight.data = torch.relu(self.weight.data)

            if self.restart_on_feasible and self.strictly_feasible_indices is not None:
                # We reset multipliers to zero when their corresponding constraint
                # is *strictly* feasible.
                # We do not reset multipliers for active constraints (satisfied with
                # equality) to avoid changing the value of a multiplier whose
                # optimal value is potentially strictly positive.

                # TODO(juan43ramirez): Document https://github.com/cooper-org/cooper/issues/28
                # about the pitfalls of using dual_restars with stateful optimizers.

                self.weight.data[self.strictly_feasible_indices, ...] = self.default_restart_value

                grad = self.weight.grad
                if grad is not None and torch.any(self.strictly_feasible_indices):
                    if grad.is_sparse:
                        indices = grad._indices()
                        values = grad._values()

                        masked_values = values * (~self.strictly_feasible_indices[indices[0]])
                        non_zero_mask = masked_values.squeeze().nonzero().squeeze()
                        non_zero_indices = indices[:, non_zero_mask]
                        non_zero_values = masked_values[non_zero_mask]

                        grad = torch.sparse_coo_tensor(non_zero_indices, non_zero_values, grad.shape)

                    else:
                        grad[self.strictly_feasible_indices, ...] = 0.0

            self.strictly_feasible_indices = None

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
    """Simplest kind of trainable Lagrange multiplier.

    :py:class:`~cooper.multipliers.DenseMultiplier`\\s are suitable for low to mid-scale
    :py:class:`~cooper.constraints.ConstraintGroup`\\s for which all the constraints
    in the group are measured constantly.

    For large-scale :py:class:`~cooper.constraints.ConstraintGroup`\\s (for example,
    one constraint per training example) you may consider using an
    :py:class:`~cooper.multipliers.IndexedMultiplier`.
    """

    def forward(self):
        """Return the current value of the multiplier."""
        return torch.clone(self.weight)

    def __repr__(self):
        return f"DenseMultiplier({self.implicit_constraint_type}, shape={self.weight.shape})"


class IndexedMultiplier(ExplicitMultiplier):
    """Indexed multipliers extend the functionality of
    :py:class:`~cooper.multipliers.DenseMultiplier`\\s to cases where the number of
    constraints in the :py:class:`~cooper.constraints.ConstraintGroup` is too large.
    This situation may arise, for example, when imposing point-wise constraints over all
    the training samples in a learning task.

    In such cases, it might be computationally prohibitive to measure the value for all
    the constraints in the :py:class:`~cooper.constraints.ConstraintGroup` and one may
    typically resort to sampling. :py:class:`~cooper.multipliers.IndexedMultiplier`\\s
    enable time-efficient retrieval of the multipliers for the sampled constraints only,
    and memory-efficient sparse gradients (on GPU).
    """

    def __init__(self, init: torch.Tensor, *args, use_sparse_gradient: bool = True, **kwargs):
        super(IndexedMultiplier, self).__init__(init=init, *args, **kwargs)
        self.last_seen_mask = torch.zeros_like(init, dtype=torch.bool)

        if use_sparse_gradient and not torch.cuda.is_available():
            warnings.warn("Backend for sparse gradients is only supported on GPU.")

        # Backend for sparse gradients only supported on GPU.
        self.use_sparse_gradient = use_sparse_gradient and torch.cuda.is_available()

    def forward(self, indices: torch.Tensor):
        """Return the current value of the multiplier at the provided indices."""

        if indices.dtype != torch.long:
            # Not allowing for boolean "indices", which are treated as indices by
            # torch.nn.functional.embedding and *not* as masks.
            raise ValueError("Indices must be of type torch.long.")

        # Mark the corresponding constraints as "seen" since the last multiplier update.
        self.last_seen_mask[indices] = True

        multiplier_values = torch.nn.functional.embedding(indices, self.weight, sparse=self.use_sparse_gradient)

        # Flatten multiplier values to 1D since Embedding works with 2D tensors.
        return torch.flatten(multiplier_values)

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """Move the multipler to a new device and/or change its dtype."""
        self.last_seen_mask = self.last_seen_mask.to(device=device)
        self.device = device
        return super().to(device=device, dtype=dtype)

    def __repr__(self):
        return f"IndexedMultiplier({self.implicit_constraint_type}, shape={self.weight.shape})"

    def post_step_(self):
        if self.strictly_feasible_indices is not None:
            # Only consider a constraint feasible if it was seen since the last step.
            feasible_mask = torch.zeros_like(self.last_seen_mask)
            feasible_mask[self.strictly_feasible_indices] = True
            feasible_filter = feasible_mask & self.last_seen_mask

            self.strictly_feasible_indices = feasible_filter

        super().post_step_()

        # Clear the contents of the seen mask.
        self.last_seen_mask *= False


class ImplicitMultiplier(Multiplier):
    """An implicit multiplier is a :py:class:`~torch.nn.Module` that computes the value
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
