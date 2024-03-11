import abc

import cooper.formulations.utils as formulation_utils
from cooper.constraints.constraint_state import ConstraintState, ConstraintStore, ConstraintType
from cooper.multipliers import Multiplier, PenaltyCoefficient, evaluate_constraint_factor


class Formulation(abc.ABC):
    # TODO(gallego-posada): Add documentation
    expects_multiplier: bool
    expects_penalty_coefficient: bool

    @abc.abstractmethod
    def __init__(self, constraint_type: ConstraintType):
        pass

    @abc.abstractmethod
    def compute_contribution_for_primal_lagrangian(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def compute_contribution_for_dual_lagrangian(self, *args, **kwargs):
        pass


class LagrangianFormulation(Formulation):
    expects_multiplier = True
    expects_penalty_coefficient = False

    def __init__(self, constraint_type: ConstraintType):
        if constraint_type not in [ConstraintType.EQUALITY, ConstraintType.INEQUALITY]:
            raise ValueError("LagrangianFormulation requires an equality or inequality constraint.")
        self.constraint_type = constraint_type

    def compute_contribution_for_primal_lagrangian(
        self, constraint_state: ConstraintState, multiplier: Multiplier
    ) -> ConstraintStore:

        if not constraint_state.contributes_to_primal_update:
            return None
        else:

            violation, strict_violation = constraint_state.extract_violations()
            constraint_features, strict_constraint_features = constraint_state.extract_constraint_features()
            multiplier_value = evaluate_constraint_factor(
                module=multiplier, violation=violation, constraint_features=constraint_features
            )
            weighted_violation = formulation_utils.compute_primal_weighted_violation(
                constraint_factor_value=multiplier_value, violation=violation
            )
            primal_constraint_store = ConstraintStore(
                multiplier_value=multiplier_value,
                violation=violation,
                lagrangian_contribution=weighted_violation,
            )

            return primal_constraint_store

    def compute_contribution_for_dual_lagrangian(
        self, constraint_state: ConstraintState, multiplier: Multiplier
    ) -> tuple[ConstraintStore, ConstraintStore]:

        if not constraint_state.contributes_to_dual_update:
            return None
        else:

            violation, strict_violation = constraint_state.extract_violations()
            constraint_features, strict_constraint_features = constraint_state.extract_constraint_features()
            multiplier_value = evaluate_constraint_factor(
                module=multiplier, violation=strict_violation, constraint_features=strict_constraint_features
            )
            weighted_violation = formulation_utils.compute_dual_weighted_violation(
                constraint_factor_value=multiplier_value, violation=strict_violation
            )
            dual_constraint_store = ConstraintStore(
                multiplier_value=multiplier_value,
                violation=strict_violation,
                lagrangian_contribution=weighted_violation,
            )

            return dual_constraint_store

    def __repr__(self):
        return f"LagrangianFormulation(constraint_type={self.constraint_type})"


class AugmentedLagrangianFormulation(Formulation):
    expects_multiplier = True
    expects_penalty_coefficient = True

    def __init__(
        self, constraint_type: ConstraintType, penalty_growth_factor: float = 1.01, violation_tolerance: float = 1e-4
    ):
        """Implements the Augmented Lagrangian formulation.

        .. warning::
            This formulation is only compatible with the
            :class:`cooper.optim.AugmentedLagrangianPrimalDualOptimizer` and
            :class:`cooper.optim.AugmentedLagrangianDualPrimalOptimizer` classes.

            The dual optimizers must all be SGD with a ``lr=1.0`` and ``maximize=True``.

        Args:
            constraint_type: Type of constraint that this formulation will be applied to.
            penalty_growth_factor: The factor by which the penalty coefficient will be
                multiplied when the constraint is violated beyond ``violation_tolerance``.
            violation_tolerance: Tolerance for the constraint violation. If the
                violation is smaller than this value, the penalty coefficient is not
                updated. The comparison is done at the constraint-level (i.e., each
                entry of the violation tensor). For equality constraints, the absolute
                violation is compared to the tolerance. All constraint types use the
                strict violation (when available) for the comparison.
        """

        self.constraint_type = constraint_type
        if constraint_type not in [ConstraintType.EQUALITY, ConstraintType.INEQUALITY]:
            raise ValueError("AugmentedLagrangianFormulation requires either an equality or inequality constraint.")

        self.penalty_growth_factor = penalty_growth_factor
        self.violation_tolerance = violation_tolerance

    def compute_contribution_for_primal_lagrangian(
        self, constraint_state: ConstraintState, multiplier: Multiplier, penalty_coefficient: PenaltyCoefficient
    ) -> ConstraintStore:

        if not constraint_state.contributes_to_primal_update:
            return None
        else:

            violation, strict_violation = constraint_state.extract_violations()
            constraint_features, strict_constraint_features = constraint_state.extract_constraint_features()
            multiplier_value = evaluate_constraint_factor(
                module=multiplier, violation=violation, constraint_features=constraint_features
            )
            penalty_coefficient_value = evaluate_constraint_factor(
                module=penalty_coefficient, violation=violation, constraint_features=constraint_features
            )

            augmented_weighted_violation = formulation_utils.compute_quadratic_augmented_contribution(
                multiplier_value=multiplier_value,
                penalty_coefficient_value=penalty_coefficient_value,
                violation=violation,
                constraint_type=self.constraint_type,
            )

            primal_constraint_store = ConstraintStore(
                lagrangian_contribution=augmented_weighted_violation,
                violation=violation,
                multiplier_value=multiplier_value,
                penalty_coefficient_value=penalty_coefficient_value,
            )

            return primal_constraint_store

    def compute_contribution_for_dual_lagrangian(
        self, constraint_state: ConstraintState, multiplier: Multiplier, penalty_coefficient: PenaltyCoefficient
    ) -> ConstraintStore:

        if not constraint_state.contributes_to_dual_update:
            return None
        else:

            violation, strict_violation = constraint_state.extract_violations()
            constraint_features, strict_constraint_features = constraint_state.extract_constraint_features()
            multiplier_value = evaluate_constraint_factor(
                module=multiplier, violation=strict_violation, constraint_features=strict_constraint_features
            )

            # TODO: why does evaluate_constraint_factor use violation instead of strict_violation?
            penalty_coefficient_value = evaluate_constraint_factor(
                module=penalty_coefficient, violation=violation, constraint_features=constraint_features
            )
            weighted_violation = formulation_utils.compute_dual_weighted_violation(
                constraint_factor_value=multiplier_value,
                violation=strict_violation,
                penalty_coefficient_value=penalty_coefficient_value,
            )
            dual_constraint_store = ConstraintStore(
                lagrangian_contribution=weighted_violation,
                violation=strict_violation,
                multiplier_value=multiplier_value,
            )

            return dual_constraint_store

    def __repr__(self):
        return f"AugmentedLagrangianFormulation(constraint_type={self.constraint_type})"
