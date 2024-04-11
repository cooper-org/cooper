import abc
from collections import namedtuple
from typing import Optional

import cooper.formulations.utils as formulation_utils
from cooper.constraints.constraint_state import ConstraintState
from cooper.constraints.constraint_type import ConstraintType
from cooper.multipliers import Multiplier, PenaltyCoefficient, evaluate_constraint_factor

ContributionStore = namedtuple(
    "ContributionStore", ["lagrangian_contribution", "multiplier_value", "penalty_coefficient_value"]
)


class Formulation(abc.ABC):
    """
    Formulations prescribe how the different constraints contribute to the primal- and
    dual-differentiable Lagrangians. In other words, they define how the constraints
    affect the gradients of the Lagrangian with respect to the primal and dual variables.
    """

    expects_multiplier: bool
    expects_penalty_coefficient: bool

    @abc.abstractmethod
    def __init__(self, constraint_type: ConstraintType):
        pass

    @abc.abstractmethod
    def compute_contribution_to_primal_lagrangian(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def compute_contribution_to_dual_lagrangian(self, *args, **kwargs):
        pass


class LagrangianFormulation(Formulation):
    expects_multiplier = True
    expects_penalty_coefficient = False

    def __init__(self, constraint_type: ConstraintType):
        if constraint_type not in [ConstraintType.EQUALITY, ConstraintType.INEQUALITY]:
            raise ValueError("LagrangianFormulation requires an equality or inequality constraint.")
        self.constraint_type = constraint_type

    def compute_contribution_to_primal_lagrangian(
        self, constraint_state: ConstraintState, multiplier: Multiplier
    ) -> Optional[ContributionStore]:

        if not constraint_state.contributes_to_primal_update:
            return None

        violation, strict_violation = constraint_state.extract_violations()
        constraint_features, strict_constraint_features = constraint_state.extract_constraint_features()

        eval_factor_kwargs = dict(constraint_features=constraint_features, expand_shape=violation.shape)
        multiplier_value = evaluate_constraint_factor(module=multiplier, **eval_factor_kwargs)

        lagrangian_contribution = formulation_utils.compute_primal_weighted_violation(
            constraint_factor_value=multiplier_value, violation=violation
        )

        return ContributionStore(lagrangian_contribution, multiplier_value, penalty_coefficient_value=None)

    def compute_contribution_to_dual_lagrangian(
        self, constraint_state: ConstraintState, multiplier: Multiplier
    ) -> Optional[ContributionStore]:

        if not constraint_state.contributes_to_dual_update:
            return None

        violation, strict_violation = constraint_state.extract_violations()
        constraint_features, strict_constraint_features = constraint_state.extract_constraint_features()

        eval_factor_kwargs = dict(constraint_features=strict_constraint_features, expand_shape=strict_violation.shape)
        multiplier_value = evaluate_constraint_factor(module=multiplier, **eval_factor_kwargs)

        lagrangian_contribution = formulation_utils.compute_dual_weighted_violation(
            multiplier_value=multiplier_value, violation=strict_violation
        )

        return ContributionStore(lagrangian_contribution, multiplier_value, penalty_coefficient_value=None)

    def __repr__(self):
        return f"LagrangianFormulation(constraint_type={self.constraint_type})"


class AugmentedLagrangianFormulation(Formulation):
    expects_multiplier = True
    expects_penalty_coefficient = True

    def __init__(self, constraint_type: ConstraintType):
        """Implements the Augmented Lagrangian formulation.

        .. warning::
            This formulation is only compatible with the
            :class:`cooper.optim.AugmentedLagrangianPrimalDualOptimizer` and
            :class:`cooper.optim.AugmentedLagrangianDualPrimalOptimizer` classes.

            The dual optimizers must all be SGD with a ``lr=1.0`` and ``maximize=True``.

        Args:
            constraint_type: Type of constraint that this formulation will be applied to.
        """

        self.constraint_type = constraint_type
        if constraint_type not in [ConstraintType.EQUALITY, ConstraintType.INEQUALITY]:
            raise ValueError("AugmentedLagrangianFormulation requires either an equality or inequality constraint.")

    def compute_contribution_to_primal_lagrangian(
        self, constraint_state: ConstraintState, multiplier: Multiplier, penalty_coefficient: PenaltyCoefficient
    ) -> Optional[ContributionStore]:

        if not constraint_state.contributes_to_primal_update:
            return None

        violation, strict_violation = constraint_state.extract_violations()
        constraint_features, strict_constraint_features = constraint_state.extract_constraint_features()

        eval_factor_kwargs = dict(constraint_features=constraint_features, expand_shape=violation.shape)
        multiplier_value = evaluate_constraint_factor(module=multiplier, **eval_factor_kwargs)
        penalty_coefficient_value = evaluate_constraint_factor(module=penalty_coefficient, **eval_factor_kwargs)

        lagrangian_contribution = formulation_utils.compute_quadratic_augmented_contribution(
            multiplier_value=multiplier_value,
            penalty_coefficient_value=penalty_coefficient_value,
            violation=violation,
            constraint_type=self.constraint_type,
        )

        return ContributionStore(lagrangian_contribution, multiplier_value, penalty_coefficient_value)

    def compute_contribution_to_dual_lagrangian(
        self, constraint_state: ConstraintState, multiplier: Multiplier, penalty_coefficient: PenaltyCoefficient
    ) -> Optional[ContributionStore]:

        if not constraint_state.contributes_to_dual_update:
            return None

        violation, strict_violation = constraint_state.extract_violations()
        constraint_features, strict_constraint_features = constraint_state.extract_constraint_features()

        eval_factor_kwargs = dict(constraint_features=strict_constraint_features, expand_shape=strict_violation.shape)
        multiplier_value = evaluate_constraint_factor(module=multiplier, **eval_factor_kwargs)
        penalty_coefficient_value = evaluate_constraint_factor(module=penalty_coefficient, **eval_factor_kwargs)

        lagrangian_contribution = formulation_utils.compute_dual_weighted_violation(
            multiplier_value=multiplier_value,
            violation=strict_violation,
            penalty_coefficient_value=penalty_coefficient_value,
        )

        return ContributionStore(lagrangian_contribution, multiplier_value, penalty_coefficient_value)

    def __repr__(self):
        return f"AugmentedLagrangianFormulation(constraint_type={self.constraint_type})"
