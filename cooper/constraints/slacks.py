import abc
import warnings

import torch


class SlackVariable(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self):
        """Return the current value of the slack variable."""
        pass

    @abc.abstractmethod
    def post_step_(self):
        """
        Post-step function for slack variables. This function is called after each step
        of the primal optimizer.
        """
        pass

    def __repr__(self):
        return f"{type(self).__name__}(shape={self.weight.shape})"


class ConstantSlack(SlackVariable):
    """Constant (non-trainable) slack variable.

    Args:
        init: Value of the slack variable.
    """

    def __init__(self, init: torch.Tensor):
        super().__init__()

        if init.requires_grad:
            raise ValueError("Constant slack should not be trainable.")
        self.weight = init
        self.device = init.device

    def forward(self):
        """Return the current value of the slacks."""
        return torch.clone(self.weight)

    def parameters(self):
        """Return an empty iterator for consistency with slack variables which are
        :py:class:`~torch.nn.Module`."""
        return iter(())

    def post_step_(self):
        # Constant slacks are not trainable and therefore do not require a post-step.
        pass

    def state_dict(self):
        return {"weight": self.weight}

    def load_state_dict(self, state_dict):
        self.weight = state_dict["weight"]


class ExplicitSlack(SlackVariable):
    """
    An explicit slack holds a :py:class:`~torch.nn.parameter.Parameter` which contains
    (explicitly) the value of the slack variable with a
    :py:class:`~cooper.constraints.ConstraintGroup` in a
    :py:class:`~cooper.cmp.ConstrainedMinimizationProblem`.

    Args:
        init: Initial value of the slack variable.
        enforce_positive: Whether to enforce non-negativity on the values of the
            slack variable.
    """

    def __init__(self, init: torch.Tensor, *, enforce_positive: bool = False):
        super().__init__()

        self.enforce_positive = enforce_positive

        self.weight = torch.nn.Parameter(init)
        if self.enforce_positive and torch.any(self.weight.data < 0):
            raise ValueError("For non-negative slack, all entries in init tensor must be non-negative.")

        self.device = self.weight.device

    def post_step_(self):
        """
        Post-step function for slack variables. This function is called after each step
        of the primal optimizer.
        """
        if self.enforce_positive:
            # Ensures non-negativity of the slack variables by projecting them onto the
            # non-negative orthant.
            self.weight.data = torch.relu(self.weight.data)

    def state_dict(self):
        _state_dict = super().state_dict()
        _state_dict["enforce_positive"] = self.enforce_positive
        return _state_dict

    def load_state_dict(self, state_dict):
        self.enforce_positive = state_dict.pop("enforce_positive")
        super().load_state_dict(state_dict)
        self.device = self.weight.device


class DenseSlack(ExplicitSlack):
    """Simplest kind of trainable slack variable.

    :py:class:`~cooper.constraints.slacks.DenseSlack`\\s are suitable for low to
    mid-scale :py:class:`~cooper.constraints.ConstraintGroup`\\s for which all the
    constraints in the group are measured constantly.

    For large-scale :py:class:`~cooper.constraints.ConstraintGroup`\\s (for example,
    one constraint per training example) you may consider using an
    :py:class:`~cooper.constraints.slacks.IndexedSlack`.
    """

    def forward(self):
        """Return the current value of the slack variables."""
        return torch.clone(self.weight)


class IndexedSlack(ExplicitSlack):
    """Indexed slacks extend the functionality of
    :py:class:`~cooper.constraints.slacks.DenseSlack`\\s to cases where the number of
    constraints in the :py:class:`~cooper.constraints.ConstraintGroup` is too large.
    This situation may arise, for example, when imposing point-wise constraints over all
    the training samples in a learning task.

    In such cases, it might be computationally prohibitive to measure the value for all
    the constraints in the :py:class:`~cooper.constraints.ConstraintGroup` and one may
    typically resort to sampling. :py:class:`~cooper.constraints.slacks.IndexedSlack`\\s
    enable time-efficient retrieval of the slack variables for the sampled constraints
    only, and memory-efficient sparse gradients (on GPU).
    """

    def __init__(self, init: torch.Tensor, *args, use_sparse_gradient: bool = True, **kwargs):
        super(IndexedSlack, self).__init__(init=init, *args, **kwargs)

        if use_sparse_gradient and not torch.cuda.is_available():
            warnings.warn("Backend for sparse gradients is only supported on GPU.")

        # Backend for sparse gradients only supported on GPU.
        self.use_sparse_gradient = use_sparse_gradient and torch.cuda.is_available()

    def forward(self, indices: torch.Tensor):
        """Return the current value of the slack at the provided indices."""

        if indices.dtype != torch.long:
            # Not allowing for boolean "indices", which are treated as indices by
            # torch.nn.functional.embedding and *not* as masks.
            raise ValueError("Indices must be of type torch.long.")

        slack_values = torch.nn.functional.embedding(indices, self.weight, sparse=self.use_sparse_gradient)

        # Flatten slack values to 1D since Embedding works with 2D tensors.
        return torch.flatten(slack_values)
