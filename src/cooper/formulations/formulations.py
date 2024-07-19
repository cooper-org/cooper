import abc
from typing import Literal, NamedTuple, Optional

import torch

import cooper.formulations.utils as formulation_utils
from cooper.constraints.constraint_state import ConstraintState
from cooper.constraints.constraint_type import ConstraintType
from cooper.multipliers import Multiplier, PenaltyCoefficient


class ContributionStore(NamedTuple):
    lagrangian_contribution: torch.Tensor
    multiplier_value: torch.Tensor
    penalty_coefficient_value: Optional[torch.Tensor] = None


class Formulation(abc.ABC):
    """Formulations prescribe how the different constraints contribute to the primal- and
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

    def sanity_check_penalty_coefficient(self, penalty_coefficient: Optional[PenaltyCoefficient]) -> None:
        if self.expects_penalty_coefficient and penalty_coefficient is None:
            raise ValueError(f"{type(self).__name__} expects a penalty coefficient but none was provided.")
        if not self.expects_penalty_coefficient and penalty_coefficient is not None:
            raise ValueError(f"Received unexpected penalty coefficient for {type(self).__name__}.")

    def _prepare_kwargs_for_lagrangian_contribution(
        self,
        constraint_state: ConstraintState,
        multiplier: Multiplier,
        penalty_coefficient: Optional[PenaltyCoefficient],
        primal_or_dual: Literal["primal", "dual"],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        self.sanity_check_penalty_coefficient(penalty_coefficient)

        violation, strict_violation = constraint_state.extract_violations()
        constraint_features, strict_constraint_features = constraint_state.extract_constraint_features()

        if primal_or_dual == "dual":
            violation = strict_violation
            constraint_features = strict_constraint_features

        eval_factor_kwargs = dict(constraint_features=constraint_features, expand_shape=violation.shape)
        multiplier_value = formulation_utils.evaluate_constraint_factor(module=multiplier, **eval_factor_kwargs)
        penalty_coefficient_value = None
        if self.expects_penalty_coefficient:
            penalty_coefficient_value = formulation_utils.evaluate_constraint_factor(
                module=penalty_coefficient, **eval_factor_kwargs
            )

        return violation, multiplier_value, penalty_coefficient_value

    def compute_contribution_to_dual_lagrangian(
        self,
        constraint_state: ConstraintState,
        multiplier: Multiplier,
        penalty_coefficient: Optional[PenaltyCoefficient],
    ) -> Optional[ContributionStore]:
        if not constraint_state.contributes_to_dual_update:
            return None

        violation, multiplier_value, penalty_coefficient_value = self._prepare_kwargs_for_lagrangian_contribution(
            constraint_state=constraint_state,
            multiplier=multiplier,
            penalty_coefficient=penalty_coefficient,
            primal_or_dual="dual",
        )
        lagrangian_contribution = formulation_utils.compute_dual_weighted_violation(
            multiplier_value=multiplier_value, violation=violation, penalty_coefficient_value=penalty_coefficient_value
        )

        return ContributionStore(lagrangian_contribution, multiplier_value, penalty_coefficient_value)

    @abc.abstractmethod
    def compute_contribution_to_primal_lagrangian(self, *args, **kwargs):
        pass


class LagrangianFormulation(Formulation):
    expects_penalty_coefficient = False

    def compute_contribution_to_primal_lagrangian(
        self,
        constraint_state: ConstraintState,
        multiplier: Multiplier,
        penalty_coefficient: Optional[PenaltyCoefficient],
    ) -> Optional[ContributionStore]:
        if not constraint_state.contributes_to_primal_update:
            return None

        violation, multiplier_value, penalty_coefficient_value = self._prepare_kwargs_for_lagrangian_contribution(
            constraint_state=constraint_state,
            multiplier=multiplier,
            penalty_coefficient=penalty_coefficient,
            primal_or_dual="primal",
        )
        lagrangian_contribution = formulation_utils.compute_primal_weighted_violation(
            constraint_factor_value=multiplier_value, violation=violation
        )

        return ContributionStore(lagrangian_contribution, multiplier_value, penalty_coefficient_value)


class AugmentedLagrangianFormulation(Formulation):
    """Implements the Augmented Lagrangian formulation.

    .. warning::
        The dual optimizers must all be SGD with a ``lr=1.0`` and ``maximize=True``.
    """

    expects_penalty_coefficient = True

    def compute_contribution_to_primal_lagrangian(
        self,
        constraint_state: ConstraintState,
        multiplier: Multiplier,
        penalty_coefficient: Optional[PenaltyCoefficient],
    ) -> Optional[ContributionStore]:
        if not constraint_state.contributes_to_primal_update:
            return None

        violation, multiplier_value, penalty_coefficient_value = self._prepare_kwargs_for_lagrangian_contribution(
            constraint_state=constraint_state,
            multiplier=multiplier,
            penalty_coefficient=penalty_coefficient,
            primal_or_dual="primal",
        )
        lagrangian_contribution = formulation_utils.compute_primal_quadratic_augmented_contribution(
            multiplier_value=multiplier_value,
            penalty_coefficient_value=penalty_coefficient_value,
            violation=violation,
            constraint_type=self.constraint_type,
        )

        return ContributionStore(lagrangian_contribution, multiplier_value, penalty_coefficient_value)
