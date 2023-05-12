import abc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from cooper.constraints import ConstraintGroup, ConstraintState, ConstraintType, observed_constraints_iterator
from cooper.multipliers import ExplicitMultiplier

# Formulation, and some other classes below, are heavily inspired by the design of the
# TensorFlow Constrained Optimization (TFCO) library:
# https://github.com/google-research/tensorflow_constrained_optimization


@dataclass
class LagrangianStore:
    """Stores the value of the (primal) Lagrangian, the dual Lagrangian, as well as the
    values of the observed multipliers."""

    lagrangian: torch.Tensor
    dual_lagrangian: Optional[torch.Tensor] = None
    observed_multipliers: Optional[list[torch.Tensor]] = None


class CMPState:
    """Represents the state of a Constrained Minimization Problem in terms of the value
    of its loss and constraint violations/defects.

    Args:
        loss: Value of the loss or main objective to be minimized :math:`f(x)`
        observed_constraints: List of tuples containing the observed/measured constraint
            groups along with their states. The constraint state may be held internally
            by the constraint group (`constraint_group.state`), or it may be passed
            explicitly as the second element of the tuple.
        misc: Optional storage space for additional information relevant to the state of
            the CMP. This dict enables persisting the results of certain computations
            for post-processing. For example, one may want to retain the value of the
            predictions/logits computed over a given minibatch during the call to
            :py:meth:`~.ConstrainedMinimizationProblem.compute_cmp_state` to measure or
            log training statistics.
    """

    def __init__(
        self,
        loss: Optional[torch.Tensor] = None,
        observed_constraints: Union[Sequence[ConstraintGroup], Sequence[Tuple[ConstraintGroup, ConstraintState]]] = (),
        misc: Optional[dict] = None,
    ):
        self.loss = loss
        self.observed_constraints = observed_constraints
        self.misc = misc

        self._primal_lagrangian = None
        self._dual_lagrangian = None

    def populate_lagrangian(self, return_multipliers: bool = False) -> LagrangianStore:
        """Computes and accumulates the Lagrangian based on the loss and the
        contributions to the "primal" and "dual" Lagrangians resulting from each of the
        observed constraints.

        The Lagrangian contributions correspond to disjoint computational graphs from
        the point of view of gradient propagation: there is no gradient connection
        between the primal (resp. dual) Lagrangian contribution and the dual (resp.
        primal) variables.

        Args:
            return_multipliers: When `True`, we return the value of the multipliers for
                the observed constraints.

        Returns:
            primal_lagrangian: Value of the Lagrangian. This tensor has gradient with
                respect to the primal variables.
            observed_multiplier_values: When `return_multipliers=True`, return the value
                of the multiplier for each of the observed_constraints.
        """

        # Check if any of the observed constraints will contribute to the primal and
        # dual Lagrangians
        any_primal_contribution = False
        any_dual_contribution = False
        for constraint_group, constraint_state in observed_constraints_iterator(self.observed_constraints):
            if not constraint_state.skip_dual_contribution:
                any_dual_contribution = True
            if not constraint_state.skip_primal_contribution:
                any_primal_contribution = True

        if self.loss is None and not any_primal_contribution:
            # No loss provided, and no observed constraints will contribute to the
            # primal Lagrangian.
            primal_lagrangian = None
        else:
            # Either a loss was provided, or at least one observed constraint will
            # contribute to the primal Lagrangian.
            primal_lagrangian = 0.0 if self.loss is None else torch.clone(self.loss)

        dual_lagrangian = 0.0 if any_dual_contribution else None

        observed_multiplier_values = []

        for constraint_group, constraint_state in observed_constraints_iterator(self.observed_constraints):
            multiplier_value, primal_contribution, dual_contribution = constraint_group.compute_lagrangian_contribution(
                constraint_state=constraint_state
            )

            if not constraint_state.skip_primal_contribution:
                primal_lagrangian += primal_contribution
            if not constraint_state.skip_dual_contribution:
                # Determine which of the constraints are strictly feasible and update
                # the `strictly_feasible_indices` attribute of the multiplier.
                if (
                    isinstance(constraint_group.multiplier, ExplicitMultiplier)
                    and (constraint_group.constraint_type == ConstraintType.INEQUALITY)
                    and constraint_group.multiplier.restart_on_feasible
                ):
                    # `strict_violation` has already been patched to contain `violation`
                    # whenever `strict_violation` is not provided.
                    constraint_group.multiplier.strictly_feasible_indices = constraint_state.strict_violation < 0.0

                dual_lagrangian += dual_contribution

            if return_multipliers:
                observed_multiplier_values.append(multiplier_value)

        if primal_lagrangian is not None:
            # Either a loss was provided, or at least one observed constraint
            # contributed to the primal Lagrangian.
            previous_primal_lagrangian = 0.0 if self._primal_lagrangian is None else self._primal_lagrangian
            self._primal_lagrangian = primal_lagrangian + previous_primal_lagrangian

        if dual_lagrangian is not None:
            # Some observed constraints contributed to the dual Lagrangian
            previous_dual_lagrangian = 0.0 if self._dual_lagrangian is None else self._dual_lagrangian
            self._dual_lagrangian = dual_lagrangian + previous_dual_lagrangian

        lagrangian_store = LagrangianStore(lagrangian=self._primal_lagrangian, dual_lagrangian=self._dual_lagrangian)
        if return_multipliers:
            lagrangian_store.observed_multipliers = observed_multiplier_values

        return lagrangian_store

    def purge_lagrangian(self, purge_primal: bool, purge_dual: bool) -> None:
        """Purge the accumulated Lagrangian contributions."""
        if purge_primal:
            self._primal_lagrangian = None
        if purge_dual:
            self._dual_lagrangian = None

    def primal_backward(self) -> None:
        """Triggers backward calls to compute the gradient of the Lagrangian with
        respect to the primal variables."""
        if self._primal_lagrangian is not None and isinstance(self._primal_lagrangian, torch.Tensor):
            self._primal_lagrangian.backward()

        # After completing the backward call, we purge the accumulated _primal_lagrangian
        self.purge_lagrangian(purge_primal=True, purge_dual=False)

    def dual_backward(self) -> None:
        """Triggers backward calls to compute the gradient of the Lagrangian with
        respect to the dual variables."""
        if self._dual_lagrangian is not None and isinstance(self._dual_lagrangian, torch.Tensor):
            self._dual_lagrangian.backward()

        # After completing the backward call, we purge the accumulated _dual_lagrangian
        self.purge_lagrangian(purge_primal=False, purge_dual=True)

    def backward(self) -> None:
        """Computes the gradient of the Lagrangian with respect to both the primal and
        dual parameters."""
        self.primal_backward()
        self.dual_backward()

    def __repr__(self) -> str:
        _string = f"CMPState(loss={self.loss}, \n"
        for constraint_group, constraint_state in observed_constraints_iterator(self.observed_constraints):
            _string += f"  {constraint_group}: {constraint_state}, \n"
        _string += f"misc={self.misc})"
        return _string


class ConstrainedMinimizationProblem(abc.ABC):
    """Template for constrained minimization problems."""

    def __init__(self):
        self._state = CMPState()

    @property
    def state(self) -> CMPState:
        return self._state

    @state.setter
    def state(self, value: CMPState):
        self._state = value

    @abc.abstractmethod
    def compute_cmp_state(self, *args, **kwargs) -> CMPState:
        """Computes the state of the CMP based on the current value of the primal
        parameters.

        The signature of this abstract function may be changed to accommodate situations
        that require a model, (mini-batched) inputs/targets, or other arguments to be
        passed.

        Structuring the CMP class around this method, enables the re-use of shared
        sections of a computational graph. For example, consider a case where we want to
        minimize a model's cross entropy loss subject to a constraint on the entropy of
        its predictions. Both of these quantities depend on the predicted logits (on a
        minibatch). This closure-centric design allows flexible problem specifications
        while avoiding re-computation.
        """

    def compute_violations(self) -> CMPState:
        """Computes the violation of (a subset of) the constraints of the CMP based on
        the current value of the primal parameters. This function returns a
        :py:class:`cooper.problem.CMPState` collecting the values of the observed
        constraints. Note that the returned ``CMPState`` may have ``loss=None`` since,
        by design, the value of the loss is not necessarily computed when evaluating
        `only` the constraints.

        The signature of this "abstract" function may be changed to accommodate
        situations that require a model, (mini-batched) inputs/targets, or other
        arguments to be passed.

        Depending on the problem at hand, the computation of the constraints can be
        compartimentalized in a way that is independent of the evaluation of the loss.
        Alternatively, :py:meth:`~.ConstrainedMinimizationProblem.compute_violations`
        may be called during the execution of the
        :py:meth:`~.ConstrainedMinimizationProblem.compute_cmp_state` method.
        """
        raise NotImplementedError
