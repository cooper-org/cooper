import abc
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import torch

from cooper.constraints import Constraint, ConstraintState
from cooper.multipliers import Multiplier, PenaltyCoefficient


@dataclass
class LagrangianStore:
    """Stores the value of the (primal or dual) Lagrangian, as well as the multiplier
    and penalty coefficient values for the observed constraints.
    """

    lagrangian: Optional[torch.Tensor] = None
    multiplier_values: dict[Constraint, torch.Tensor] = field(default_factory=dict)
    penalty_coefficient_values: dict[Constraint, torch.Tensor] = field(default_factory=dict)

    def backward(self) -> None:
        """Triggers backward calls to compute the gradient of the Lagrangian with
        respect to the primal variables.
        """
        if self.lagrangian is not None:
            self.lagrangian.backward()

    def observed_multiplier_values(self) -> Iterator[torch.Tensor]:
        yield from self.multiplier_values.values()

    def observed_penalty_coefficient_values(self) -> Iterator[torch.Tensor]:
        yield from self.penalty_coefficient_values.values()


@dataclass
class CMPState:
    """Represents the state of a :py:class:`~.ConstrainedMinimizationProblem` in terms of
    the value of its loss and constraint violations.

    Args:
        loss: Value of the loss or main objective to be minimized :math:`f(x)`
        observed_constraints: Dictionary with :py:class:`~Constraint` instances as keys
            and :py:class:`~ConstraintState` instances as values.
        misc: Optional storage space for additional information relevant to the state of
            the CMP. This dict enables persisting the results of certain computations
            for post-processing. For example, one may want to retain the value of the
            predictions/logits computed over a given minibatch during the call to
            :py:meth:`~.ConstrainedMinimizationProblem.compute_cmp_state` to measure or
            log training statistics.
    """

    loss: Optional[torch.Tensor] = None
    observed_constraints: dict[Constraint, ConstraintState] = field(default_factory=dict)
    misc: Optional[dict] = None

    def _compute_primal_or_dual_lagrangian(self, primal_or_dual: Literal["primal", "dual"]) -> LagrangianStore:
        """Computes the primal or dual Lagrangian based on the loss and the
        contribution of the observed constraints.

        We don't count the loss towards the dual Lagrangian since the objective is not
        a function of the dual variables.
        """
        check_contributes_fn = lambda cs: getattr(cs, f"contributes_to_{primal_or_dual}_update")
        contributing_constraints = {c: cs for c, cs in self.observed_constraints.items() if check_contributes_fn(cs)}

        if not contributing_constraints:
            # No observed constraints contribute to the Lagrangian.
            lagrangian = self.loss.clone() if primal_or_dual == "primal" and self.loss is not None else None
            return LagrangianStore(lagrangian=lagrangian)

        lagrangian = self.loss.clone() if primal_or_dual == "primal" and self.loss is not None else 0.0
        multiplier_values = {}
        penalty_coefficient_values = {}

        for constraint, constraint_state in contributing_constraints.items():
            contribution_store = constraint.compute_contribution_to_lagrangian(constraint_state, primal_or_dual)
            if contribution_store is not None:
                lagrangian = lagrangian + contribution_store.lagrangian_contribution
                multiplier_values[constraint] = contribution_store.multiplier_value
                if contribution_store.penalty_coefficient_value is not None:
                    penalty_coefficient_values[constraint] = contribution_store.penalty_coefficient_value

        return LagrangianStore(
            lagrangian=lagrangian,
            multiplier_values=multiplier_values,
            penalty_coefficient_values=penalty_coefficient_values,
        )

    def compute_primal_lagrangian(self) -> LagrangianStore:
        """Computes and accumulates the primal-differentiable Lagrangian based on the
        loss and the contribution of the observed constraints.
        """
        return self._compute_primal_or_dual_lagrangian(primal_or_dual="primal")

    def compute_dual_lagrangian(self) -> LagrangianStore:
        """Computes and accumulates the dual-differentiable Lagrangian based on the
        contribution of the observed constraints.

        The dual Lagrangian contained in ``LagrangianStore.lagrangian`` ignores the
        contribution of the loss, since the objective function does not depend on the
        dual variables. Therefore, ``LagrangianStore.lagrangian = 0`` regardless of
        the value of ``self.loss``.
        """
        return self._compute_primal_or_dual_lagrangian(primal_or_dual="dual")

    def observed_violations(self) -> Iterator[torch.Tensor]:
        for constraint_state in self.observed_constraints.values():
            yield constraint_state.violation

    def observed_strict_violations(self) -> Iterator[torch.Tensor]:
        for constraint_state in self.observed_constraints.values():
            yield constraint_state.strict_violation

    def observed_constraint_features(self) -> Iterator[torch.Tensor]:
        for constraint_state in self.observed_constraints.values():
            yield constraint_state.constraint_features

    def observed_strict_constraint_features(self) -> Iterator[torch.Tensor]:
        for constraint_state in self.observed_constraints.values():
            yield constraint_state.strict_constraint_features


class ConstrainedMinimizationProblem(abc.ABC):
    """Template for constrained minimization problems, where subclasses represent specific
    constrained optimization problems.

    Subclasses must override the :py:meth:`~.ConstrainedMinimizationProblem.compute_cmp_state`
    method. This method should return a :py:class:`~.CMPState` instance that encapsulates the
    current state of the optimization problem, including the evaluated loss and the values of
    the constraint violations.
    """

    def __init__(self) -> None:
        self._constraints = OrderedDict()

    def _register_constraint(self, name: str, constraint: Constraint) -> None:
        """Registers a constraint with the CMP.

        Args:
            name: Name of the constraint.
            constraint: Constraint instance to be registered.
        """
        if not isinstance(constraint, Constraint):
            raise TypeError(f"Expected a Constraint instance, got {type(constraint)}")
        if name in self._constraints:
            raise ValueError(f"Constraint with name {name} already exists")

        self._constraints[name] = constraint

    def constraints(self) -> Iterator[Constraint]:
        """Return an iterator over the registered constraints of the CMP."""
        yield from self._constraints.values()

    def named_constraints(self) -> Iterator[tuple[str, Constraint]]:
        """Return an iterator over the registered constraints of the CMP, yielding
        tuples of the form ``(constraint_name, constraint)``.
        """
        yield from self._constraints.items()

    def multipliers(self) -> Iterator[Multiplier]:
        """Returns an iterator over the multipliers associated with the registered
        constraints of the CMP.
        """
        for constraint in self.constraints():
            yield constraint.multiplier

    def named_multipliers(self) -> Iterator[tuple[str, Multiplier]]:
        """Returns an iterator over the multipliers associated with the registered
        constraints of the CMP, yielding tuples of the form ``(constraint_name, multiplier)``.
        """
        for constraint_name, constraint in self.named_constraints():
            yield constraint_name, constraint.multiplier

    def penalty_coefficients(self) -> Iterator[PenaltyCoefficient]:
        """Returns an iterator over the penalty coefficients associated with the
        registered constraints of the CMP. Constraints without penalty coefficients
        are skipped.
        """
        for constraint in self.constraints():
            if constraint.penalty_coefficient is not None:
                yield constraint.penalty_coefficient

    def named_penalty_coefficients(self) -> Iterator[tuple[str, PenaltyCoefficient]]:
        """Returns an iterator over the penalty coefficients associated with the
        registered  constraints of the CMP, yielding tuples of the form
        ``(constraint_name, penalty_coefficient)``. Constraints without penalty
        coefficients are skipped.
        """
        for constraint_name, constraint in self.named_constraints():
            if constraint.penalty_coefficient is not None:
                yield constraint_name, constraint.penalty_coefficient

    def dual_parameters(self) -> Iterator[torch.nn.Parameter]:
        """Return an iterator over the parameters of the multipliers associated with the
        registered constraints of the CMP. This method is useful for instantiating the
        dual optimizers. If a multiplier is shared by several constraints, we only
        return its parameters once.
        """
        for multiplier in {constraint.multiplier for constraint in self.constraints()}:
            yield from multiplier.parameters()

    def to_(self, *args: str, **kwargs: str) -> None:
        # TODO: document, test
        for constraint in self.constraints():
            constraint.multiplier = constraint.multiplier.to(*args, **kwargs)

    def state_dict(self) -> dict:
        """Returns the state of the CMP. This includes the state of the multipliers and penalty coefficients."""
        state_dict = {
            "multipliers": {name: multiplier.state_dict() for name, multiplier in self.named_multipliers()},
            "penalty_coefficients": {name: pc.state_dict() for name, pc in self.named_penalty_coefficients()},
        }
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """Loads the state of the CMP. This includes the state of the multipliers and penalty coefficients.

        Args:
            state_dict: A state dictionary containing the state of the CMP.
        """
        for name, multiplier_state_dict in state_dict["multipliers"].items():
            self._constraints[name].multiplier.load_state_dict(multiplier_state_dict)
            self._constraints[name].multiplier.sanity_check()
        for name, penalty_coefficient_state_dict in state_dict["penalty_coefficients"].items():
            self._constraints[name].penalty_coefficient.load_state_dict(penalty_coefficient_state_dict)

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Constraint):
            self._register_constraint(name, value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        if name in self._constraints:
            return self._constraints[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __delattr__(self, name: str) -> None:
        if name in self._constraints:
            del self._constraints[name]
        else:
            super().__delattr__(name)

    def __repr__(self) -> str:
        repr_str = f"{type(self).__name__}"
        if len(self._constraints) < 5:
            repr_str += "\n\t(constraints=[\n"
            for i, (name, constraint) in enumerate(self.named_constraints()):
                suffix = ",\n" if i < len(self._constraints) - 1 else "\n"
                repr_str += f"\t\t{name}: {constraint}{suffix}"
            repr_str += "\t\t]\n\t)"
        return repr_str

    @abc.abstractmethod
    def compute_cmp_state(self, *args: Any, **kwargs: Any) -> CMPState:
        """Computes the state of the CMP based on the current value of the primal
        parameters.

        The signature of this function may be adjusted to accommodate situations
        that require a model, (mini-batched) inputs/targets, or other arguments to be
        passed.
        """

    def sanity_check_cmp_state(self, cmp_state: CMPState) -> None:
        """Performs sanity checks on the CMP state.

        This helper method is useful for ensuring that the CMP state is well-formed.
        """
        if cmp_state.loss is not None and cmp_state.loss.grad is None:
            raise ValueError("The loss tensor must have a valid gradient.")

        for constraint, constraint_state in cmp_state.observed_constraints.items():
            if constraint_state.violation.grad is None:
                raise ValueError(f"The violation tensor of constraint {constraint} must have a valid gradient.")
            if constraint_state.strict_violation.grad is not None:
                raise ValueError(
                    f"The strict violation tensor of constraint {constraint} has a non-null gradient: "
                    f"{constraint_state.strict_violation.grad}."
                )

    def compute_violations(self, *args: Any, **kwargs: Any) -> CMPState:
        """Computes the violation of the CMP constraints based on the current value of the
        primal parameters. This function returns a :py:class:`~.CMPState` instance
        containing the observed constraint values. Note that the returned
        :py:class:`~.CMPState` may have ``loss=None``, as the loss value is not
        necessarily computed when only evaluating the constraints.

        The function signature may be adjusted to accommodate situations that require a
        model, (mini-batched) inputs/targets, or other arguments.

        In some cases, the computation of constraints may be independent of loss
        evaluation. In such situations,
        :py:meth:`~.ConstrainedMinimizationProblem.compute_violations` can be called
        as part of the execution of
        :py:meth:`~.ConstrainedMinimizationProblem.compute_cmp_state`.
        """
        raise NotImplementedError
