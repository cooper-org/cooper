"""Classes for modeling dual variables (e.g. Lagrange multipliers)."""

import abc
from typing import Any, Optional

import torch

from cooper.utils import ConstraintType


class Multiplier(torch.nn.Module, abc.ABC):
    expects_constraint_features: bool
    constraint_type: ConstraintType

    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Return the current value of the multiplier."""

    @abc.abstractmethod
    def post_step_(self) -> None:
        """Post-step function for multipliers. This function is called after each step of
        the dual optimizer, and allows for additional post-processing of the implicit
        multiplier module or its parameters.
        """

    def sanity_check(self) -> None:
        """Perform sanity checks on the multiplier. This method is called after setting
        the constraint type and ensures consistency between the multiplier and the
        constraint type. For example, multipliers for inequality constraints must be
        non-negative.
        """

    def set_constraint_type(self, constraint_type: ConstraintType) -> None:
        self.constraint_type = constraint_type
        self.sanity_check()


class ExplicitMultiplier(Multiplier):
    """An ExplicitMultiplier holds a :py:class:`torch.nn.parameter.Parameter` (`weight`)
    which explicitly contains the value of the Lagrange multipliers associated with a
    :py:class:`~cooper.constraints.Constraint` in a
    :py:class:`~cooper.cmp.ConstrainedMinimizationProblem`.

    Args:
        num_constraints: Number of constraints associated with the multiplier.
        init: Tensor used to initialize the multiplier values. If both ``init`` and
            ``num_constraints`` are provided, ``init`` must have shape ``(num_constraints,)``.
        device: Device for the multiplier. If ``None``, the device is inferred from the
            ``init`` tensor or the default device.
        dtype: Data type for the multiplier. Default is ``torch.float32``.
    """

    def __init__(
        self,
        num_constraints: Optional[int] = None,
        init: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.weight = self.initialize_weight(num_constraints=num_constraints, init=init, device=device, dtype=dtype)

    @staticmethod
    def initialize_weight(
        num_constraints: Optional[int],
        init: Optional[torch.Tensor],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Initialize the weight of the multiplier. If both ``init`` and
        ``num_constraints`` are provided (and the shapes are consistent), ``init`` takes
        precedence. Otherwise, the weight is initialized to :py:func:`torch.zeros` of
        shape ``(num_constraints,)``.

        Raises:
            ValueError: If both ``num_constraints`` and ``init`` are ``None``.
            ValueError: If both ``num_constraints`` and ``init`` are provided but
                their shapes are inconsistent.
            ValueError: If the provided ``init`` is not a 1D tensor.
        """
        if num_constraints is None and init is None:
            raise ValueError("At least one of `num_constraints` and `init` must be provided.")
        if num_constraints is not None and init is not None and num_constraints != init.shape[0]:
            raise ValueError(f"Inconsistent `init` shape {init.shape} and `num_constraints={num_constraints}")

        if init is not None:
            if init.dim() != 1:
                raise ValueError("`init` must be a 1D tensor of shape `(num_constraints,)`.")
            return torch.nn.Parameter(init.to(device=device, dtype=dtype))

        return torch.nn.Parameter(torch.zeros(num_constraints, device=device, dtype=dtype))

    @property
    def device(self) -> torch.device:
        return self.weight.device

    def sanity_check(self) -> None:
        """Ensures multipliers for inequality constraints are non-negative.

        Raises:
            ValueError: If the multiplier is associated with an inequality constraint
                and any of its entries is negative.
        """
        if self.constraint_type == ConstraintType.INEQUALITY and torch.any(self.weight.data < 0):
            raise ValueError("For inequality constraint, all entries in multiplier must be non-negative.")

    @torch.no_grad()
    def post_step_(self) -> None:
        """Projects (in-place) multipliers associated with inequality constraints so
        that they remain non-negative. This function is called after each dual optimizer
        step.
        """
        if self.constraint_type == ConstraintType.INEQUALITY:
            # Ensures non-negativity for multipliers associated with inequality constraints.
            self.weight.data = torch.relu(self.weight.data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(num_constraints={self.weight.shape[0]})"


class DenseMultiplier(ExplicitMultiplier):
    r"""Sub-class of :py:class:`~cooper.multipliers.ExplicitMultiplier` for constraints
    that are all evaluated at every optimization step.
    """

    expects_constraint_features = False

    def forward(self) -> torch.Tensor:
        """Returns the current value of the multiplier."""
        return torch.clone(self.weight)


class IndexedMultiplier(ExplicitMultiplier):
    r""":py:class:`~cooper.multipliers.ExplicitMultiplier` for indexed constraints which
    are evaluated only for a subset of constraints on every optimization step.
    """

    expects_constraint_features = True

    def __init__(
        self,
        num_constraints: Optional[int] = None,
        init: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(num_constraints, init, device, dtype)
        if self.weight.dim() == 1:
            # To use the forward call in F.embedding, we must reshape the weight to be a
            # 2-dim tensor
            self.weight.data = self.weight.data.unsqueeze(-1)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Return the current value of the multiplier at the provided indices.

        Args:
            indices: Indices of the multipliers to return. The shape of ``indices`` must
                be ``(num_indices,)``.

        Raises:
            ValueError: If ``indices`` dtype is not ``torch.long``.
        """
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
    """An implicit multiplier is a :py:class:`torch.nn.Module` that computes the value
    of a Lagrange multiplier associated with a
    :py:class:`~cooper.constraints.Constraint` based on the "features" for each
    constraint. The multiplier is *implicitly* represented by its parameters.
    """

    @abc.abstractmethod
    def forward(self) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def post_step_(self) -> None:
        """This method is called after each step of the dual optimizer and allows for
        additional post-processing of the implicit multiplier module or its parameters.
        """
