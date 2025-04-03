import abc
from typing import Any, Literal, NamedTuple, Optional

import torch

import cooper.formulations.utils as formulation_utils
from cooper.constraints.constraint_state import ConstraintState
from cooper.multipliers import Multiplier
from cooper.penalty_coefficients import PenaltyCoefficient
from cooper.utils import ConstraintType


class ContributionStore(NamedTuple):
    lagrangian_contribution: torch.Tensor
    multiplier_value: Optional[torch.Tensor] = None
    penalty_coefficient_value: Optional[torch.Tensor] = None


class Formulation(abc.ABC):
    """Formulations prescribe how the different constraints contribute to the primal- and
    dual-differentiable Lagrangians. In other words, they prescribe how the constraints
    affect the gradients of the Lagrangian with respect to the primal and dual variables.

    Attributes:
        expects_multiplier (bool): Used to determine whether the formulation requires a
            multiplier.
        expects_penalty_coefficient (bool): Used to determine whether the formulation
            requires a penalty coefficient.

    Raises:
        ValueError: If the constraint type is not equality or inequality.
    """

    expects_multiplier: bool
    expects_penalty_coefficient: bool

    def __init__(self, constraint_type: ConstraintType) -> None:
        if constraint_type not in {ConstraintType.EQUALITY, ConstraintType.INEQUALITY}:
            raise ValueError(f"{type(self).__name__} requires either an equality or inequality constraint.")
        self.constraint_type = constraint_type

    def __repr__(self) -> str:
        return f"{type(self).__name__}(constraint_type={self.constraint_type})"

    def sanity_check_multiplier(self, multiplier: Optional[Multiplier]) -> None:
        """Ensures that the multiplier is provided if and only if it is expected.

        Raises:
            ValueError: If a multiplier is expected but not provided, or vice versa.
        """
        if self.expects_multiplier and multiplier is None:
            raise ValueError(f"{type(self).__name__} expects a multiplier but none was provided.")
        if not self.expects_multiplier and multiplier is not None:
            raise ValueError(f"Received unexpected multiplier for {type(self).__name__}.")

    def sanity_check_penalty_coefficient(self, penalty_coefficient: Optional[PenaltyCoefficient]) -> None:
        """Ensures that the penalty is provided if and only if it is expected.

        Raises:
            ValueError: If a penalty coefficient is expected but not provided, or vice versa.
        """
        if self.expects_penalty_coefficient and penalty_coefficient is None:
            raise ValueError(f"{type(self).__name__} expects a penalty coefficient but none was provided.")
        if not self.expects_penalty_coefficient and penalty_coefficient is not None:
            raise ValueError(f"Received unexpected penalty coefficient for {type(self).__name__}.")

    def _prepare_kwargs_for_lagrangian_contribution(
        self,
        constraint_state: ConstraintState,
        multiplier: Optional[Multiplier],
        penalty_coefficient: Optional[PenaltyCoefficient],
        primal_or_dual: Literal["primal", "dual"],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prepares the arguments for the computation of the Lagrangian contribution.

        Depending on the chosen formulation, the contribution of a constraint to the
        Lagrangian would require different inputs. This method processes a
        :py:class:`cooper.constraints.constraint_state.ConstraintState` and prepares
        the necessary information to compute the contribution to the Lagrangian.

        This method extracts and patches the constraint violations and features depending
        on the information available. For example, no `strict_violation` is provided,
        the call to `extract_violations` will return the same tensor for both `violation`
        and `strict_violation` (with the latter tensor having been detached).

        It also evaluates the constraint factors (multiplier and penalty coefficient
        modules) using the constraint features (if provided).

        Args:
            constraint_state: The :py:class:`cooper.constraints.constraint_state.ConstraintState`
                object.
            multiplier: The multiplier module.
            penalty_coefficient: The penalty coefficient module.
            primal_or_dual: If `"primal"`, we prepare the arguments to compute the
                primal-differentiable contribution to the Lagrangian. Analogous for
                the case of `"dual"`.

        Returns:
            A tuple containing the following objects:

            If `primal_or_dual == "primal"`:
                - violation: The observed constraint violations tensor.
                - multiplier_value: The evaluated multiplier factor.
        """
        violation, strict_violation = constraint_state.extract_violations()
        constraint_features, strict_constraint_features = constraint_state.extract_constraint_features()

        if primal_or_dual == "dual":
            violation = strict_violation
            constraint_features = strict_constraint_features

        eval_factor_kwargs = {"constraint_features": constraint_features, "expand_shape": violation.shape}

        multiplier_value = None
        if self.expects_multiplier:
            multiplier_value = formulation_utils.evaluate_constraint_factor(module=multiplier, **eval_factor_kwargs)

        penalty_coefficient_value = None
        if self.expects_penalty_coefficient:
            penalty_coefficient_value = formulation_utils.evaluate_constraint_factor(
                module=penalty_coefficient, **eval_factor_kwargs
            )

        return violation, multiplier_value, penalty_coefficient_value

    @abc.abstractmethod
    def compute_contribution_to_primal_lagrangian(self, *args: Any, **kwargs: Any) -> Optional[ContributionStore]:
        """Computes the contribution of a given constraint violation to the *primal*
        Lagrangian.

        Returns ``None`` if the constraint does not contribute to the primal update
        (i.e., when ``ConstraintState.contributes_to_primal_update=False``).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_contribution_to_dual_lagrangian(self, *args: Any, **kwargs: Any) -> Optional[ContributionStore]:
        """Computes the contribution of a given constraint violation to the *dual*
        Lagrangian.

        Returns ``None`` if the constraint does not contribute to the dual update
        (i.e., when ``ConstraintState.contributes_to_dual_update=False``).
        """
        raise NotImplementedError


class Lagrangian(Formulation):
    r"""The Lagrangian formulation implements the following primal Lagrangian:

    .. math::
        \Lag_{\text{primal}}(\vx, \vlambda, \vmu) = f(\vx) + \vlambda^{\top} \tilde{\vg}(\vx) + \vmu^{\top} \tilde{\vh}(\vx).

    And the following dual Lagrangian:

    .. math::
        \Lag_{\text{dual}}(\vx, \vlambda, \vmu) = \vlambda^{\top} \vg(\vx) + \vmu^{\top} \vh(\vx).
    """

    expects_multiplier = True
    expects_penalty_coefficient = False

    def compute_contribution_to_primal_lagrangian(
        self, constraint_state: ConstraintState, multiplier: Multiplier
    ) -> Optional[ContributionStore]:
        if not constraint_state.contributes_to_primal_update:
            return None

        # Third return is `penalty_coefficient_value` which is always `None` for this formulation.
        violation, multiplier_value, _ = self._prepare_kwargs_for_lagrangian_contribution(
            constraint_state=constraint_state, multiplier=multiplier, penalty_coefficient=None, primal_or_dual="primal"
        )
        lagrangian_contribution = formulation_utils.compute_primal_weighted_violation(
            constraint_factor_value=multiplier_value, violation=violation
        )

        return ContributionStore(lagrangian_contribution, multiplier_value, None)

    def compute_contribution_to_dual_lagrangian(
        self, constraint_state: ConstraintState, multiplier: Multiplier
    ) -> Optional[ContributionStore]:
        if not constraint_state.contributes_to_dual_update:
            return None

        # Third return is `penalty_coefficient_value` which is always `None` for this formulation.
        violation, multiplier_value, _ = self._prepare_kwargs_for_lagrangian_contribution(
            constraint_state=constraint_state, multiplier=multiplier, penalty_coefficient=None, primal_or_dual="dual"
        )
        lagrangian_contribution = formulation_utils.compute_dual_weighted_violation(
            multiplier_value=multiplier_value, violation=violation
        )

        return ContributionStore(lagrangian_contribution, multiplier_value, None)


class QuadraticPenalty(Formulation):
    r"""The Quadratic Penalty formulation implements the following primal Lagrangian:

    .. math::
        \Lag_{\text{primal}}(\vx) = f(\vx) + \frac{1}{2} \vc_{\vg}^\top \,
        \texttt{relu}(\tilde{\vg}(\vx))^2 + \frac{1}{2} \vc_{\vh}^\top \,
        \tilde{\vh}(\vx)^2.

    It does not implement a dual Lagrangian since it does not consider dual variables.
    """

    expects_multiplier = False
    expects_penalty_coefficient = True

    def compute_contribution_to_primal_lagrangian(
        self, constraint_state: ConstraintState, penalty_coefficient: PenaltyCoefficient
    ) -> Optional[ContributionStore]:
        if not constraint_state.contributes_to_primal_update:
            return None

        # Second return is `multiplier_value` which is always `None` for this formulation.
        violation, _, penalty_coefficient_value = self._prepare_kwargs_for_lagrangian_contribution(
            constraint_state=constraint_state,
            multiplier=None,
            penalty_coefficient=penalty_coefficient,
            primal_or_dual="primal",
        )

        lagrangian_contribution = formulation_utils.compute_quadratic_penalty(
            penalty_coefficient_value=penalty_coefficient_value,
            violation=violation,
            constraint_type=self.constraint_type,
        )

        return ContributionStore(lagrangian_contribution, None, penalty_coefficient_value)

    def compute_contribution_to_dual_lagrangian(  # noqa: PLR6301
        self,
        constraint_state: ConstraintState,  # noqa: ARG002
        penalty_coefficient: PenaltyCoefficient,  # noqa: ARG002
    ) -> None:
        """The Quadratic Penalty formulation does not involve dual variables and
        therefore does not implement a dual Lagrangian (returns ``None``).
        """
        return


class AugmentedLagrangian(Formulation):
    r"""The Augmented Lagrangian formulation implements the following primal Lagrangian:

    .. math::
        \Lag_{\text{primal}}(\vx, \vlambda, \vmu) = f(\vx) + \vlambda^{\top}
        \tilde{\vg}(\vx) + \vmu^{\top} \tilde{\vh}(\vx) + \frac{1}{2} \vc_{\vg}^\top \,
        \texttt{relu}(\tilde{\vg}(\vx))^2 + \frac{1}{2} \vc_{\vh}^\top \,
        \tilde{\vh}(\vx)^2.

    And the following dual Lagrangian:

    .. math::
        \Lag_{\text{dual}}(\vx, \vlambda, \vmu) = \vlambda^{\top} \vg(\vx) + \vmu^{\top} \vh(\vx).
    """

    expects_multiplier = True
    expects_penalty_coefficient = True

    def compute_contribution_to_primal_lagrangian(
        self, constraint_state: ConstraintState, multiplier: Multiplier, penalty_coefficient: PenaltyCoefficient
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

    def compute_contribution_to_dual_lagrangian(
        self, constraint_state: ConstraintState, multiplier: Multiplier, penalty_coefficient: PenaltyCoefficient
    ) -> Optional[ContributionStore]:
        if not constraint_state.contributes_to_dual_update:
            return None

        violation, multiplier_value, penalty_coefficient_value = self._prepare_kwargs_for_lagrangian_contribution(
            constraint_state=constraint_state,
            multiplier=multiplier,
            penalty_coefficient=penalty_coefficient,
            primal_or_dual="dual",
        )

        # Not providing a penalty coefficient since the dual Lagrangian is just the
        # sum of the violation times the multiplier.
        lagrangian_contribution = formulation_utils.compute_dual_weighted_violation(
            multiplier_value=multiplier_value, violation=violation
        )

        return ContributionStore(lagrangian_contribution, multiplier_value, penalty_coefficient_value)
