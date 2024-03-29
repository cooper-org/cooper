import abc
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

import torch

from cooper.constraints import Constraint, ConstraintMeasurement, ConstraintState

# Formulation, and some other classes below, are inspired by the design of the
# TensorFlow Constrained Optimization (TFCO) library:
# https://github.com/google-research/tensorflow_constrained_optimization


@dataclass
class LagrangianStore:
    """Stores the value of the (primal) Lagrangian, the dual Lagrangian, as well as
    ``ConstraintMeasurement``s for the primal-contributing and dual-contributing
    constraints."""

    lagrangian: Optional[torch.Tensor] = None
    dual_lagrangian: Optional[torch.Tensor] = None
    primal_constraint_measurements: list[ConstraintMeasurement] = field(default_factory=list)
    dual_constraint_measurements: list[ConstraintMeasurement] = field(default_factory=list)

    def primal_backward(self) -> None:
        """Triggers backward calls to compute the gradient of the Lagrangian with
        respect to the primal variables."""
        if self.lagrangian is not None and isinstance(self.lagrangian, torch.Tensor):
            self.lagrangian.backward()

        # After completing the backward call, we purge the accumulated lagrangian
        self._purge_primal_lagrangian()

    def dual_backward(self) -> None:
        """Triggers backward calls to compute the gradient of the Lagrangian with
        respect to the dual variables."""
        if self.dual_lagrangian is not None and isinstance(self.dual_lagrangian, torch.Tensor):
            self.dual_lagrangian.backward()

        # After completing the backward call, we purge the accumulated dual_lagrangian
        self._purge_dual_lagrangian()

    def backward(self) -> None:
        """Computes the gradient of the Lagrangian with respect to both the primal and
        dual parameters."""
        self.primal_backward()
        self.dual_backward()

    def _purge_primal_lagrangian(self) -> None:
        """Purge the accumulated primal Lagrangian contributions."""
        self.lagrangian = None
        self.primal_constraint_measurements = []

    def _purge_dual_lagrangian(self) -> None:
        """Purge the accumulated dual Lagrangian contributions."""
        self.dual_lagrangian = None
        self.dual_constraint_measurements = []

    def multiplier_values_for_primal_constraints(self):
        return [_.multiplier_value for _ in self.primal_constraint_measurements]

    def multiplier_values_for_dual_constraints(self):
        return [_.multiplier_value for _ in self.dual_constraint_measurements]


@dataclass
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

    loss: Optional[torch.Tensor] = None
    observed_constraints: Sequence[tuple[Constraint, ConstraintState]] = ()
    misc: Optional[dict] = None


class ConstrainedMinimizationProblem(abc.ABC):
    """Template for constrained minimization problems."""

    def __init__(self):
        self.lagrangian_store = LagrangianStore()

    def populate_primal_lagrangian_(self, cmp_state: CMPState) -> LagrangianStore:
        """Computes and accumulates the primal-differentiable Lagrangian based on the
        loss and the contribution of the observed constraints.
        """

        contributing_constraints = [
            (cg, cs) for cg, cs in cmp_state.observed_constraints if cs.contributes_to_primal_update
        ]

        if cmp_state.loss is None and len(contributing_constraints) == 0:
            # No loss provided, and no observed constraints will contribute to the
            # primal Lagrangian.
            # We return any existent values for the private attributes, unmodified.
            return self.lagrangian_store

        # Either a loss was provided, or at least one observed constraint contributes to
        # the primal Lagrangian.
        previous_primal_lagrangian = (
            0.0 if self.lagrangian_store.lagrangian is None else self.lagrangian_store.lagrangian
        )
        current_primal_lagrangian = 0.0 if cmp_state.loss is None else torch.clone(cmp_state.loss)

        current_primal_constraint_measurements = []
        for constraint, constraint_state in contributing_constraints:
            primal_constraint_contribution_out = constraint.compute_constraint_primal_contribution(constraint_state)
            primal_lagrangian_contribution, primal_constraint_measurement = primal_constraint_contribution_out
            current_primal_constraint_measurements.append(primal_constraint_measurement)
            if primal_lagrangian_contribution is not None:
                current_primal_lagrangian = current_primal_lagrangian + primal_lagrangian_contribution

        # Modify "private" attributes to accumulate Lagrangian values over successive
        # calls to `populate_primal_lagrangian`
        self.lagrangian_store.lagrangian = previous_primal_lagrangian + current_primal_lagrangian
        self.lagrangian_store.primal_constraint_measurements.extend(current_primal_constraint_measurements)

        # We return any existent values for the _dual_lagrangian, and the
        # _dual_constraint_measurements. The _primal_lagrangian and _primal_constraint_measurements
        # attributes have been modified earlier, so their updated values are returned.
        return self.lagrangian_store

    def populate_dual_lagrangian_(self, cmp_state: CMPState) -> LagrangianStore:
        """Computes and accumulates the dual-differentiable Lagrangian based on the
        loss and the contribution of the observed constraints.
        """
        contributing_constraints = [
            (cg, cs) for cg, cs in cmp_state.observed_constraints if cs.contributes_to_dual_update
        ]

        if len(contributing_constraints) == 0:
            # No observed constraints will contribute to the dual Lagrangian.
            # We return any existent values for the private attributes, unmodified.
            return self.lagrangian_store

        # At least one observed constraint contributes to the dual Lagrangian.
        previous_dual_lagrangian = (
            0.0 if self.lagrangian_store.dual_lagrangian is None else self.lagrangian_store.dual_lagrangian
        )
        current_dual_lagrangian = 0.0

        current_dual_constraint_measurements = []
        for constraint, constraint_state in contributing_constraints:
            dual_constraint_contribution_out = constraint.compute_constraint_dual_contribution(constraint_state)
            dual_lagrangian_contribution, dual_constraint_measurement = dual_constraint_contribution_out
            current_dual_constraint_measurements.append(dual_constraint_measurement)
            if dual_lagrangian_contribution is not None:
                current_dual_lagrangian = current_dual_lagrangian + dual_lagrangian_contribution

                # Extracting the violation from the dual_constraint_measurement ensures that it is
                # the "strict" violation, if available.
                _, strict_constraint_features = constraint_state.extract_constraint_features()
                constraint.update_strictly_feasible_indices_(
                    strict_violation=dual_constraint_measurement.violation,
                    strict_constraint_features=strict_constraint_features,
                )

        # Modify "private" attributes to accumulate Lagrangian values over successive
        # calls to `populate_dual_lagrangian`
        self.lagrangian_store.dual_lagrangian = previous_dual_lagrangian + current_dual_lagrangian
        self.lagrangian_store.dual_constraint_measurements.extend(current_dual_constraint_measurements)

        # We return any existent values for the _primal_lagrangian, and the
        # _primal_constraint_measurements. The _dual_lagrangian and _dual_constraint_measurements
        # attributes have been modified earlier, so their updated values are returned.
        return self.lagrangian_store

    def populate_lagrangian_(self, cmp_state: CMPState) -> LagrangianStore:
        """Computes and accumulates the Lagrangian based on the loss and the
        contributions to the "primal" and "dual" Lagrangians resulting from each of the
        observed constraints.

        The Lagrangian contributions correspond to disjoint computational graphs from
        the point of view of gradient propagation: there is no gradient connection
        between the primal (resp. dual) Lagrangian contribution and the dual (resp.
        primal) variables.

        Returns:
            lagrangian_store: LagrangianStore containing the value of the
            primal-differentiable Lagrangian, the dual-differentiable Lagrangian,
            as well as the ConstraintStores for the primal- and dual-contributing
            constraints.
        """

        # The attributes of the lagrangian_store returned by this function are populated
        # _sequentially_ and disjointly by each of the function calls below. The order
        # of the calls is not important.
        _ = self.populate_primal_lagrangian_(cmp_state)
        lagrangian_store = self.populate_dual_lagrangian_(cmp_state)
        return lagrangian_store

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

    def compute_violations(self, *args, **kwargs) -> CMPState:
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
