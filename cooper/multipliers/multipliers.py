"""Classes for modeling dual variables (e.g. Lagrange multipliers)."""

import abc
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
    :py:class:`~cooper.constraints.Constraint` in a
    :py:class:`~cooper.cmp.ConstrainedMinimizationProblem`.

    .. warning::
        When `restart_on_feasible=True`, the entries of the multiplier which correspond
        to feasible constraints in the :py:class:`~cooper.constraints.Constraint`
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
            inequality constraints (i.e. enforce_positive=True). Note that we discourage
            the use of `restart_on_feasible` along with stateful optimizers (such as
            :py:class:`~torch.optim.SGD` with momentum or :py:class:`~torch.optim.Adam`).

    """

    def __init__(
        self,
        constraint_type: ConstraintType,
        num_constraints: Optional[int] = None,
        init: Optional[torch.Tensor] = None,
        restart_on_feasible: bool = False,
        default_restart_value: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        if constraint_type not in [ConstraintType.EQUALITY, ConstraintType.INEQUALITY]:
            raise ValueError(f"Constraint type {constraint_type} is not valid to create a multiplier.")
        self.constraint_type = constraint_type
        self.enforce_positive = self.constraint_type == ConstraintType.INEQUALITY

        self.weight = self.initialize_weight(num_constraints=num_constraints, init=init, device=device, dtype=dtype)

        self.restart_on_feasible = restart_on_feasible
        self.default_restart_value = default_restart_value

        self.strictly_feasible_indices = None

        self.base_sanity_checks()

    def initialize_weight(
        self,
        num_constraints: Optional[int],
        init: Optional[torch.Tensor],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Initialize the weight of the multiplier. If both `init` and `num_constraints`
        are provided (and the shapes are consistent), `init` takes precedence.
        Otherwise, the weight is initialized to zero.
        """
        if (num_constraints is None) and (init is None):
            raise ValueError("At least one of `num_constraints` and `init` must be provided.")
        elif (num_constraints is not None) and (init is not None) and (num_constraints != init.shape[0]):
            raise ValueError(f"Inconsistent `init` shape {init.shape} and `num_constraints={num_constraints}")
        elif init is not None:
            return torch.nn.Parameter(init.to(device=device, dtype=dtype))
        elif num_constraints is not None:
            return torch.nn.Parameter(torch.zeros(num_constraints, device=device, dtype=dtype))

    @property
    def device(self):
        return self.weight.device

    def base_sanity_checks(self):
        if self.enforce_positive and torch.any(self.weight.data < 0):
            raise ValueError("For inequality constraint, all entries in multiplier must be non-negative.")

        if not self.enforce_positive and self.restart_on_feasible:
            raise ValueError("Restart on feasible is not supported for equality constraints.")

        if (self.default_restart_value < 0) and self.restart_on_feasible:
            raise ValueError("Default restart value must be positive.")

        if (self.default_restart_value > 0) and not self.restart_on_feasible:
            raise ValueError("Default restart value was provided but `restart_on_feasible=False`.")

    def post_step_(self):
        """
        Post-step function for multipliers. This function is called after each step of
        the dual optimizer, and ensures that (if required) the multipliers are
        non-negative. It also restarts the value of the multipliers for inequality
        constraints that are strictly feasible.

        # TODO(juan43ramirez): Document https://github.com/cooper-org/cooper/issues/28
        # about the pitfalls of using dual_restars with stateful optimizers.

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
                self.weight.data[self.strictly_feasible_indices, ...] = self.default_restart_value

            self.strictly_feasible_indices = None

    def state_dict(self):
        _state_dict = super().state_dict()
        _state_dict["constraint_type"] = self.constraint_type
        _state_dict["restart_on_feasible"] = self.restart_on_feasible
        _state_dict["default_restart_value"] = self.default_restart_value
        return _state_dict

    def load_state_dict(self, state_dict):
        self.constraint_type = state_dict.pop("constraint_type")
        self.restart_on_feasible = state_dict.pop("restart_on_feasible")
        self.default_restart_value = state_dict.pop("default_restart_value")
        super().load_state_dict(state_dict)


class DenseMultiplier(ExplicitMultiplier):
    """Simplest kind of trainable Lagrange multiplier.

    :py:class:`~cooper.multipliers.DenseMultiplier`\\s are suitable for low to mid-scale
    :py:class:`~cooper.constraints.Constraint`\\s for which all the constraints
    in the group are measured constantly.

    For large-scale :py:class:`~cooper.constraints.Constraint`\\s (for example,
    one constraint per training example) you may consider using an
    :py:class:`~cooper.multipliers.IndexedMultiplier`.
    """

    def forward(self):
        """Return the current value of the multiplier."""
        return torch.clone(self.weight)

    def __repr__(self):
        return f"DenseMultiplier(enforce_positive={self.enforce_positive}, shape={tuple(self.weight.shape)})"


class IndexedMultiplier(ExplicitMultiplier):
    """Indexed multipliers extend the functionality of
    :py:class:`~cooper.multipliers.DenseMultiplier`\\s to cases where the number of
    constraints in the :py:class:`~cooper.constraints.Constraint` is too large.
    This situation may arise, for example, when imposing point-wise constraints over all
    the training samples in a learning task.

    In such cases, it might be computationally prohibitive to measure the value for all
    the constraints in the :py:class:`~cooper.constraints.Constraint` and one may
    typically resort to sampling. :py:class:`~cooper.multipliers.IndexedMultiplier`\\s
    enable time-efficient retrieval of the multipliers for the sampled constraints only,
    and memory-efficient sparse gradients (on GPU).
    """

    def __init__(self, *args, **kwargs):
        super(IndexedMultiplier, self).__init__(*args, **kwargs)
        if len(self.weight.shape) == 1:
            # Must reshape weight to be a order-2 tensor to use in F.embedding forward
            self.weight.data = self.weight.data.unsqueeze(-1)
        self.last_seen_mask = torch.zeros_like(self.weight, dtype=torch.bool)

    def forward(self, indices: torch.Tensor):
        """Return the current value of the multiplier at the provided indices."""

        if indices.dtype != torch.long:
            # Not allowing for boolean "indices", which are treated as indices by
            # torch.nn.functional.embedding and *not* as masks.
            raise ValueError("Indices must be of type torch.long.")

        # Mark the corresponding constraints as "seen" since the last multiplier update.
        self.last_seen_mask[indices] = True

        # TODO(gallego-posada): Document sparse gradients are expected for stateful
        # optimizers (having buffers)
        multiplier_values = torch.nn.functional.embedding(indices, self.weight, sparse=True)

        # Flatten multiplier values to 1D since Embedding works with 2D tensors.
        return torch.flatten(multiplier_values)

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """Move the multipler to a new device and/or change its dtype."""
        self.last_seen_mask = self.last_seen_mask.to(device=device)
        return super().to(device=device, dtype=dtype)

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

    def __repr__(self):
        return f"IndexedMultiplier({self.constraint_type}, shape={self.weight.shape})"


class ImplicitMultiplier(Multiplier):
    """An implicit multiplier is a :py:class:`~torch.nn.Module` that computes the value
    of a Lagrange multiplier associated with a
    :py:class:`~cooper.constraints.Constraint` based on "features" for each
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
