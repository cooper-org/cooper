import abc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from cooper.constraints import ConstraintGroup, ConstraintState, ConstraintType
from cooper.multipliers import ExplicitMultiplier, IndexedMultiplier

# Formulation, and some other classes below, are heavily inspired by the design of the
# TensorFlow Constrained Optimization (TFCO) library:
# https://github.com/google-research/tensorflow_constrained_optimization


@dataclass
class LagrangianStore:
    """Stores the value of the (primal) Lagrangian, the dual Lagrangian, as well as the
    values of the observed multipliers."""

    lagrangian: torch.Tensor
    dual_lagrangian: Optional[torch.Tensor] = None
    # TODO: change primal_observed_multipliers and dual_observed_multipliers to
    # constraint_stores.
    primal_observed_multipliers: Optional[list[torch.Tensor]] = None
    dual_observed_multipliers: Optional[list[torch.Tensor]] = None


class CMPState:
    """Represents the state of a Constrained Minimization Problem in terms of the value
    of its loss and constraint violations/defects.

    Args:
        loss: Value of the loss or main objective to be minimized :math:`f(x)`
        observed_constraints: List of tuples containing the observed/measured constraint
            groups along with their states.
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
        observed_constraints: Sequence[Tuple[ConstraintGroup, ConstraintState]] = (),
        misc: Optional[dict] = None,
    ):
        self.loss = loss
        self.observed_constraints = observed_constraints
        self.misc = misc

        self._primal_lagrangian = None
        self._dual_lagrangian = None

    def populate_lagrangian(self, return_multipliers: bool = False) -> LagrangianStore:
        # TODO: this function could return the ConstraintStores for each of the
        # observed constraints. As such, change the return_multipliers argument to
        # return_constraint_stores.

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

        # TODO: could populate the primal and dual lagrangians separately. This is useful
        # for alternating updates to not waste computation.

        # Check if any of the observed constraints will contribute to the primal and
        # dual Lagrangians
        any_primal_contribution = any([cs.contributes_to_primal_update for cg, cs in self.observed_constraints])
        any_dual_contribution = any([cs.contributes_to_dual_update for cg, cs in self.observed_constraints])

        if self.loss is None and not any_primal_contribution:
            # No loss provided, and no observed constraints will contribute to the
            # primal Lagrangian.
            primal_lagrangian = None
        else:
            # Either a loss was provided, or at least one observed constraint will
            # contribute to the primal Lagrangian.
            primal_lagrangian = 0.0 if self.loss is None else torch.clone(self.loss)

        dual_lagrangian = 0.0 if any_dual_contribution else None

        primal_observed_multiplier_values = []
        dual_observed_multiplier_values = []

        for constraint_group, constraint_state in self.observed_constraints:
            # TODO (juan43ramirez): rename primal_store and dual_store to primal_constraint_store and dual_constraint_store
            primal_store, dual_store = constraint_group.compute_constraint_contribution(constraint_state)

            if constraint_state.contributes_to_primal_update and primal_store is not None:
                primal_lagrangian = primal_lagrangian + primal_store.lagrangian_contribution

            if constraint_state.contributes_to_dual_update and dual_store is not None:
                dual_lagrangian = dual_lagrangian + dual_store.lagrangian_contribution

                # TODO(gallego-posada): Consider forbidding the use of extrapolation and
                # restart_on_feasible together.

                # Determine which of the constraints are strictly feasible and update
                # the `strictly_feasible_indices` attribute of the multiplier.
                if (
                    isinstance(constraint_group.multiplier, ExplicitMultiplier)
                    and (constraint_group.constraint_type == ConstraintType.INEQUALITY)
                    and constraint_group.multiplier.restart_on_feasible
                ):
                    strict_violation = dual_store.violation

                    if isinstance(constraint_group.multiplier, IndexedMultiplier):
                        # Need to expand the indices to the size of the multiplier
                        strictly_feasible_indices = torch.zeros_like(
                            constraint_group.multiplier.weight, dtype=torch.bool
                        )

                        # FIXME(gallego-posada): This should be using
                        # `strict_constraint_features` instead of `constraint_features`.
                        # IndexedMultipliers have a shape of (-, 1). We need to unsqueeze
                        # dimension 1 of the violations
                        strictly_feasible_indices[constraint_state.constraint_features] = (
                            strict_violation.unsqueeze(1) < 0.0
                        )
                    else:
                        strictly_feasible_indices = strict_violation < 0.0

                    constraint_group.multiplier.strictly_feasible_indices = strictly_feasible_indices

            if return_multipliers:
                if primal_store is not None:
                    primal_observed_multiplier_values.append(primal_store.multiplier_value)
                if dual_store is not None:
                    dual_observed_multiplier_values.append(dual_store.multiplier_value)

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
            lagrangian_store.primal_observed_multipliers = primal_observed_multiplier_values
            lagrangian_store.dual_observed_multipliers = dual_observed_multiplier_values

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
        for constraint_group, constraint_state in self.observed_constraints:
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
