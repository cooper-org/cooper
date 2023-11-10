import abc

import torch

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
    def compute_lagrangian_contributions(
        self, constraint_state: ConstraintState, **kwargs
    ) -> tuple[ConstraintStore, ConstraintStore]:
        """Computes the contribution from the current constraint to the primal and dual
        Lagrangians, and evaluates the associated Lagrange multiplier or penalty
        coefficient."""
        pass

    def state_dict(self):
        return {"constraint_type": self.constraint_type}

    def load_state_dict(self, state_dict: dict):
        self.constraint_type = state_dict["constraint_type"]
        pass


class PenaltyFormulation(Formulation):
    expects_multiplier = False
    expects_penalty_coefficient = True

    def __init__(self, constraint_type: ConstraintType):
        if constraint_type != ConstraintType.PENALTY:
            raise ValueError("PenaltyFormulation expects `constraint_type=ConstraintType.PENALTY`.")
        self.constraint_type = constraint_type

    def compute_contribution_for_primal_lagrangian(
        self, constraint_state: ConstraintState, penalty_coefficient: PenaltyCoefficient
    ) -> ConstraintStore:
        if not constraint_state.contributes_to_primal_update:
            return None
        else:

            violation, strict_violation = constraint_state.extract_violations()
            constraint_features, strict_constraint_features = constraint_state.extract_constraint_features()
            penalty_coefficient_value = evaluate_constraint_factor(
                module=penalty_coefficient, violation=violation, constraint_features=constraint_features
            )
            weighted_violation = formulation_utils.compute_primal_weighted_violation(
                constraint_factor_value=penalty_coefficient_value, violation=violation
            )
            primal_constraint_store = ConstraintStore(
                violation=violation,
                penalty_coefficient_value=penalty_coefficient_value,
                lagrangian_contribution=weighted_violation,
            )

            return primal_constraint_store

    def compute_contribution_for_dual_lagrangian(self, *args, **kwargs):
        return None

    def compute_lagrangian_contributions(
        self, constraint_state: ConstraintState, penalty_coefficient: PenaltyCoefficient
    ) -> tuple[ConstraintStore, ConstraintStore]:
        primal_constraint_store = self.compute_contribution_for_primal_lagrangian(constraint_state, penalty_coefficient)
        # `compute_contribution_for_primal_lagrangian` is a dummy method for this formulation
        dual_constraint_store = None
        return primal_constraint_store, dual_constraint_store

    def __repr__(self):
        return f"PenaltyFormulation(constraint_type={self.constraint_type})"


class QuadraticPenaltyFormulation(Formulation):
    # TODO(juan43ramirez): emphasize the difference with respect to the PenaltyFormulation

    expects_multiplier = False
    expects_penalty_coefficient = True

    def __init__(self, constraint_type: ConstraintType):
        # TODO(juan43ramirez): Add documentation

        if constraint_type not in [ConstraintType.EQUALITY, ConstraintType.INEQUALITY]:
            raise ValueError("QuadraticPenaltyFormulation requires an equality or inequality constraint.")
        self.constraint_type = constraint_type

    def compute_contribution_for_primal_lagrangian(
        self, constraint_state: ConstraintState, penalty_coefficient: PenaltyCoefficient
    ) -> ConstraintStore:

        if not constraint_state.contributes_to_primal_update:
            return None
        else:

            violation, strict_violation = constraint_state.extract_violations()
            constraint_features, strict_constraint_features = constraint_state.extract_constraint_features()
            penalty_coefficient_value = evaluate_constraint_factor(
                module=penalty_coefficient, violation=violation, constraint_features=constraint_features
            )
            weighted_violation = formulation_utils.compute_quadratic_penalty(
                penalty_coefficient_value=penalty_coefficient_value,
                violation=violation,
                strict_violation=strict_violation,
                constraint_type=self.constraint_type,
            )

            primal_constraint_store = ConstraintStore(
                violation=violation,
                penalty_coefficient_value=penalty_coefficient_value,
                lagrangian_contribution=weighted_violation,
            )

            return primal_constraint_store

    def compute_contribution_for_dual_lagrangian(self, *args, **kwargs):
        return None

    def compute_lagrangian_contributions(
        self, constraint_state: ConstraintState, penalty_coefficient: PenaltyCoefficient
    ) -> tuple[ConstraintStore, ConstraintStore]:
        primal_constraint_store = self.compute_contribution_for_primal_lagrangian(constraint_state, penalty_coefficient)
        # `compute_contribution_for_primal_lagrangian` is a dummy method for this formulation
        dual_constraint_store = None
        return primal_constraint_store, dual_constraint_store

    def __repr__(self):
        return f"QuadraticPenaltyFormulation(constraint_type={self.constraint_type})"


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
                violation=weighted_violation,
                lagrangian_contribution=weighted_violation,
            )

            return dual_constraint_store

    def compute_lagrangian_contributions(
        self, constraint_state: ConstraintState, multiplier: Multiplier
    ) -> tuple[ConstraintStore, ConstraintStore]:

        # TODO(gallego-posada): Avoid extract from constraint state twice in primal and dual
        primal_constraint_store = self.compute_contribution_for_primal_lagrangian(constraint_state, multiplier)
        dual_constraint_store = self.compute_contribution_for_dual_lagrangian(constraint_state, multiplier)
        return primal_constraint_store, dual_constraint_store

    def __repr__(self):
        return f"LagrangianFormulation(constraint_type={self.constraint_type})"


class AugmentedLagrangianFormulation(Formulation):
    expects_multiplier = True
    expects_penalty_coefficient = True

    def __init__(self, constraint_type: ConstraintType):
        # TODO(juan43ramirez): Add documentation

        self.constraint_type = constraint_type
        if constraint_type not in [ConstraintType.EQUALITY, ConstraintType.INEQUALITY]:
            raise ValueError("AugmentedLagrangianFormulation requires either an equality or inequality constraint.")

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

            weighted_violation = formulation_utils.compute_primal_weighted_violation(
                constraint_factor_value=multiplier_value, violation=violation
            )
            if weighted_violation is not None and not torch.all(penalty_coefficient_value == 0):
                quadratic_penalty = formulation_utils.compute_quadratic_penalty(
                    penalty_coefficient_value=penalty_coefficient_value,
                    violation=violation,
                    strict_violation=strict_violation,
                    constraint_type=self.constraint_type,
                )
                weighted_violation += quadratic_penalty

            primal_constraint_store = ConstraintStore(
                lagrangian_contribution=weighted_violation,
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
            # Since this is the contribution to the dual-differentiable Lagrangian,
            # we skip the calculation of the penalty coefficient as they (and the primal
            # variables influencing the constraints) do not depend on the dual variables.

            weighted_violation = formulation_utils.compute_dual_weighted_violation(
                constraint_factor_value=multiplier_value, violation=strict_violation
            )
            dual_constraint_store = ConstraintStore(
                lagrangian_contribution=weighted_violation,
                violation=strict_violation,
                multiplier_value=multiplier_value,
            )

            return dual_constraint_store

    def compute_lagrangian_contributions(
        self, constraint_state: ConstraintState, multiplier: Multiplier, penalty_coefficient: PenaltyCoefficient
    ) -> tuple[ConstraintStore, ConstraintStore]:

        kwargs = dict(constraint_state=constraint_state, multiplier=multiplier, penalty_coefficient=penalty_coefficient)
        primal_constraint_store = self.compute_contribution_for_primal_lagrangian(**kwargs)
        dual_constraint_store = self.compute_contribution_for_dual_lagrangian(**kwargs)

        return primal_constraint_store, dual_constraint_store

    def __repr__(self):
        return f"AugmentedLagrangianFormulation(constraint_type={self.constraint_type})"
