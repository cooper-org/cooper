from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ConstraintState:
    r"""State of a constraint, including the current constraint violation.

    Args:
        violation: The measurement of the constraint violation at some value of the primal
            parameters. This is expected to be differentiable with respect to the primal parameters.
        constraint_features: The features of the observed (differentiable) constraint violations,
            used to evaluate the associated Lagrange multiplier. For example:

            - An :py:class:`~cooper.multipliers.IndexedMultiplier` expects the indices of the constraints
              whose Lagrange multipliers are to be retrieved.

            - An :py:class:`~cooper.multipliers.ImplicitMultiplier` expects tensor-valued features for
              the constraints, and can be used to measure the violation of a subset of the constraints within a
              :py:class:`~cooper.constraints.Constraint`.

            This field can also be used with an :py:class:`~cooper.multipliers.IndexedMultiplier`
            to measure the violation of only a subset of the constraints within a :py:class:`~cooper.constraints.Constraint`.
        strict_violation: The measurement of the constraint violation used to update the dual variables.
            If not provided, the ``violation`` is used to update the dual variables instead.
        strict_constraint_features: The features of the (possibly non-differentiable) constraint. For more
            details, see ``constraint_features``. If not provided, the ``constraint_features`` are used instead.
            ``strict_violation`` is expected when ``strict_constraint_features`` are provided.
        contributes_to_primal_update: If ``False``, the current observed constraint violation does not contribute
            to the **primal** Lagrangian but still contributes to the **dual** Lagrangian. This means the violations
            affect the update for the dual variables but not the primal variables.
        contributes_to_dual_update: If ``False``, the current observed constraint violation does not contribute
            to the **dual** Lagrangian but still contributes to the **primal** Lagrangian. This allows for less frequent
            updates to the dual variables (e.g., after several primal steps), affecting the update for the primal variables
            but not the dual variables. When ``True``, **Cooper** will update the dual variables using
            ``strict_violation`` if provided, or ``violation`` otherwise.
    """

    violation: torch.Tensor
    constraint_features: Optional[torch.Tensor] = None
    strict_violation: Optional[torch.Tensor] = None
    strict_constraint_features: Optional[torch.Tensor] = None
    contributes_to_primal_update: bool = True
    contributes_to_dual_update: bool = True

    def __post_init__(self) -> None:
        """Checks that the constraint state is well-formed.

        Raises:
            ValueError: If `strict_constraint_features` are provided, but `strict_violation` is not.
        """
        if self.strict_constraint_features is not None and self.strict_violation is None:
            raise ValueError("`strict_violation` must be provided if `strict_constraint_features` is provided.")

    def extract_violations(self, do_unsqueeze: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts the violation and strict violation from the constraint state as a
        tuple. If strict violations are not provided, this function returns
        `violation, violation`. If `do_unsqueeze` is set to `True`, this function also
        unsqueezes the violation tensors to ensure thay have at least 1-dimension.
        """
        violation = self.violation

        strict_violation = self.strict_violation if self.strict_violation is not None else self.violation

        if do_unsqueeze:
            # If the violation is a scalar, we unsqueeze it to ensure that it has at
            # least one dimension. This is important since we use einsum to compute the
            # contribution of the constraint to the Lagrangian.
            if violation.dim() == 0:
                violation = violation.unsqueeze(0)
            if strict_violation.dim() == 0:
                strict_violation = strict_violation.unsqueeze(0)

        return violation, strict_violation

    def extract_constraint_features(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts the constraint features from the constraint state as a tuple.
        If strict constraint features are not provided, this function returns
        `constraint_features, constraint_features`.
        """
        constraint_features = self.constraint_features

        if self.strict_constraint_features is not None:
            strict_constraint_features = self.strict_constraint_features
        else:
            strict_constraint_features = self.constraint_features

        return constraint_features, strict_constraint_features
