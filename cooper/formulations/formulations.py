from typing import Literal, NamedTuple, Optional

import torch

import cooper.formulations.utils as formulation_utils
from cooper.constraints.constraint_state import ConstraintState
from cooper.constraints.constraint_type import ConstraintType
from cooper.multipliers import Multiplier, PenaltyCoefficient, evaluate_constraint_factor


class ContributionStore(NamedTuple):
    lagrangian_contribution: torch.Tensor
    multiplier_value: torch.Tensor
    penalty_coefficient_value: Optional[torch.Tensor] = None


class Formulation:
    """
    Formulations prescribe how the different constraints contribute to the primal- and
    dual-differentiable Lagrangians. In other words, they define how the constraints
    affect the gradients of the Lagrangian with respect to the primal and dual variables.
    """

    expects_penalty_coefficient: bool

    def __init__(self, constraint_type: ConstraintType):
        if constraint_type not in [ConstraintType.EQUALITY, ConstraintType.INEQUALITY]:
            raise ValueError(f"{type(self).__name__} requires either an equality or inequality constraint.")
        self.constraint_type = constraint_type

    def __repr__(self):
        return f"{type(self).__name__}(constraint_type={self.constraint_type})"

    def compute_contribution_to_lagrangian(
        self,
        primal_or_dual: Literal["primal", "dual"],
        constraint_state: ConstraintState,
        multiplier: Multiplier,
        penalty_coefficient: Optional[PenaltyCoefficient],
    ) -> Optional[ContributionStore]:

        if self.expects_penalty_coefficient and penalty_coefficient is None:
            raise ValueError(f"{type(self).__name__} expects a penalty coefficient but none was provided.")

        if not getattr(constraint_state, f"contributes_to_{primal_or_dual}_update"):
            return None

        violation, strict_violation = constraint_state.extract_violations()
        constraint_features, strict_constraint_features = constraint_state.extract_constraint_features()

        if primal_or_dual == "dual":
            violation = strict_violation
            constraint_features = strict_constraint_features

        eval_factor_kwargs = dict(constraint_features=constraint_features, expand_shape=violation.shape)
        multiplier_value = evaluate_constraint_factor(module=multiplier, **eval_factor_kwargs)
        penalty_coefficient_value = None
        if self.expects_penalty_coefficient:
            penalty_coefficient_value = evaluate_constraint_factor(module=penalty_coefficient, **eval_factor_kwargs)

        if primal_or_dual == "dual":
            compute_fn = formulation_utils.compute_dual_weighted_violation
            compute_kwargs = dict(
                multiplier_value=multiplier_value,
                violation=violation,
                penalty_coefficient_value=penalty_coefficient_value,
            )
        else:
            if self.expects_penalty_coefficient:
                compute_fn = formulation_utils.compute_quadratic_augmented_contribution
                compute_kwargs = dict(
                    multiplier_value=multiplier_value,
                    penalty_coefficient_value=penalty_coefficient_value,
                    violation=violation,
                    constraint_type=self.constraint_type,
                )
            else:
                compute_fn = formulation_utils.compute_primal_weighted_violation
                compute_kwargs = dict(constraint_factor_value=multiplier_value, violation=violation)

        lagrangian_contribution = compute_fn(**compute_kwargs)

        return ContributionStore(lagrangian_contribution, multiplier_value, penalty_coefficient_value)


class LagrangianFormulation(Formulation):
    expects_penalty_coefficient = False


class AugmentedLagrangianFormulation(Formulation):
    """Implements the Augmented Lagrangian formulation.

    .. warning::
        The dual optimizers must all be SGD with a ``lr=1.0`` and ``maximize=True``.
    """

    expects_penalty_coefficient = True
