"""Classes for modeling dual variables (e.g. Lagrange multipliers)."""

import abc
from typing import Optional

import torch

from cooper.constraints.constraint_type import ConstraintType


class Multiplier(torch.nn.Module, abc.ABC):
    expects_constraint_features: bool
    constraint_type: ConstraintType

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
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

    def sanity_check(self):
        # TODO(gallego-posada): Add docstring
        pass

    def set_constraint_type(self, constraint_type):
        self.constraint_type = constraint_type
        self.sanity_check()


class ExplicitMultiplier(Multiplier):
    """
    An explicit multiplier holds a :py:class:`~torch.nn.parameter.Parameter` which
    contains (explicitly) the value of the Lagrange multipliers associated with a
    :py:class:`~cooper.constraints.Constraint` in a
    :py:class:`~cooper.cmp.ConstrainedMinimizationProblem`.

    Args:
        num_constraints: Number of constraints associated with the multiplier. This
            argument is mutually exclusive with `init`.
        init: Tensor used to initialize the multiplier values. This argument is mutually
            exclusive with `num_constraints`. If provided, the shape of `init` must be
            `(num_constraints,)`.

    """

    def __init__(
        self,
        num_constraints: Optional[int] = None,
        init: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.weight = self.initialize_weight(num_constraints=num_constraints, init=init, device=device, dtype=dtype)

    def initialize_weight(
        self,
        num_constraints: Optional[int],
        init: Optional[torch.Tensor],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Initialize the weight of the multiplier. If both `init` and `num_constraints`
        are provided (and the shapes are consistent), `init` takes precedence.
        Otherwise, the weight is initialized to zeros of shape `(num_constraints,)`.
        """

        if (num_constraints is None) and (init is None):
            raise ValueError("At least one of `num_constraints` and `init` must be provided.")
        elif (num_constraints is not None) and (init is not None) and (num_constraints != init.shape[0]):
            raise ValueError(f"Inconsistent `init` shape {init.shape} and `num_constraints={num_constraints}")
        elif init is not None:
            assert init.dim() == 1, "init must be a 1D tensor of shape `(num_constraints,)`."
            return torch.nn.Parameter(init.to(device=device, dtype=dtype))
        elif num_constraints is not None:
            return torch.nn.Parameter(torch.zeros(num_constraints, device=device, dtype=dtype))

    @property
    def device(self):
        return self.weight.device

    def sanity_check(self):
        if self.constraint_type == ConstraintType.INEQUALITY and torch.any(self.weight.data < 0):
            raise ValueError("For inequality constraint, all entries in multiplier must be non-negative.")

    @torch.no_grad()
    def post_step_(self):
        """
        Post-step function for multipliers. This function is called after each step of
        the dual optimizer, and ensures that (if required) the multipliers are
        non-negative.
        """
        if self.constraint_type == ConstraintType.INEQUALITY:
            # Ensures non-negativity for multipliers associated with inequality constraints.
            self.weight.data = torch.relu(self.weight.data)

    def __repr__(self):
        return f"{type(self).__name__}(num_constraints={self.weight.shape[0]})"


class DenseMultiplier(ExplicitMultiplier):
    """Simplest kind of trainable Lagrange multiplier.

    :py:class:`~cooper.multipliers.DenseMultiplier`\\s are suitable for low to mid-scale
    :py:class:`~cooper.constraints.Constraint`\\s for which all the constraints
    in the group are measured constantly.

    For large-scale :py:class:`~cooper.constraints.Constraint`\\s (for example,
    one constraint per training example) you may consider using an
    :py:class:`~cooper.multipliers.IndexedMultiplier`.
    """

    expects_constraint_features = False

    def forward(self):
        """Return the current value of the multiplier."""
        return torch.clone(self.weight)


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

    expects_constraint_features = True

    def __init__(self, num_constraints=None, init=None, device=None, dtype=torch.float32):
        super().__init__(num_constraints, init, device, dtype)
        if self.weight.dim() == 1:
            # To use the forward call in F.embedding, we must reshape the weight to be a
            # 2-dim tensor
            self.weight.data = self.weight.data.unsqueeze(-1)

    def forward(self, indices: torch.Tensor):
        """Return the current value of the multiplier at the provided indices."""

        if indices.dtype != torch.long:
            # Not allowing for boolean "indices", which are treated as indices by
            # torch.nn.functional.embedding and *not* as masks.
            raise ValueError("Indices must be of type torch.long.")

        # TODO(gallego-posada): Document sparse gradients are expected for stateful
        # optimizers (having buffers)
        multiplier_values = torch.nn.functional.embedding(indices, self.weight, sparse=True)

        # Flatten multiplier values to 1D since Embedding works with 2D tensors.
        return torch.flatten(multiplier_values)


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

    @abc.abstractmethod
    def post_step_(self):
        """
        This method is called after each step of the dual optimizer and allows for
        additional post-processing of the implicit multiplier module or its parameters.
        For example, one may want to enforce non-negativity of the parameters of the
        implicit multiplier. Given the high flexibility of implicit multipliers, the
        post-step function is left to be implemented by the user.
        """
        pass
