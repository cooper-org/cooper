import abc
from typing import Iterable, List, Optional, Tuple

import torch

from cooper.constraints import ConstraintGroup, ConstraintState

# Formulation, and some other classes below, are heavily inspired by the design
# of the TensorFlow Constrained Optimization (TFCO) library :
# https://github.com/google-research/tensorflow_constrained_optimization


class CMPState:
    """Represents the "state" of a Constrained Minimization Problem in terms of
    the value of its loss and constraint violations/defects.

    Args:
        loss: Value of the loss or main objective to be minimized :math:`f(x)`
        observed_constraints: List of tuples containing the observed/measured
            constraint groups along with their states. The constraint state may be
            held internally by the constraint group (`constraint_group.state`), or it
            may be passed explicitly as the second element of the tuple.
        misc: Optional additional information to be store along with the state of the CMP
    """

    def __init__(
        self,
        loss: Optional[torch.Tensor] = None,
        observed_constraints: Optional[Iterable[Tuple[ConstraintGroup, Optional[ConstraintState]]]] = (),
        misc: Optional[dict] = None,
    ):

        self.loss = loss
        self.observed_constraints = observed_constraints
        self.misc = misc

        self._primal_lagrangian = None
        self._dual_lagrangian = None

    def populate_lagrangian(
        self, return_multipliers: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Computes and accumulates the Lagrangian based on the loss and the contributions
        to the "primal" and "dual" Lagrangians resulting from each of the observed
        constraints. Recall that the Lagrangian contributions correspond to disjoint
        computational graphs from the point of view of gradient propagation: there is no
        gradient connection between the primal (resp. dual) Lagrangian contribution and
        the dual (resp. primal) variables.

        Args:
            return_multipliers: When `True`, we return the value of the multipliers for
                the observed constraints.

        """

        primal_lagrangian = self.loss
        dual_lagrangian = 0.0

        if return_multipliers:
            observed_multiplier_values = []

        for constraint_tuple in self.observed_constraints:

            if isinstance(constraint_tuple, ConstraintGroup):
                constraint_group = constraint_tuple
                constraint_state = constraint_group.state
            elif isinstance(constraint_tuple, tuple) and len(constraint_tuple) == 2:
                constraint_group, constraint_state = constraint_tuple
            else:
                raise ValueError(
                    f"Received invalid format for observed constraint. Expected {ConstraintGroup} or {Tuple[ConstraintGroup, ConstraintState]}, but received {type(constraint_tuple)}"
                )

            multiplier_value, primal_contribution, dual_contribution = constraint_group.compute_lagrangian_contribution(
                constraint_state=constraint_state
            )

            primal_lagrangian += primal_contribution
            dual_lagrangian += dual_contribution

            if return_multipliers:
                observed_multiplier_values.append(multiplier_value)

        previous_primal_lagrangian = 0.0 if self._primal_lagrangian is None else self._primal_lagrangian
        self._primal_lagrangian = primal_lagrangian + previous_primal_lagrangian

        previous_dual_lagrangian = 0.0 if self._dual_lagrangian is None else self._dual_lagrangian
        self._dual_lagrangian = dual_lagrangian + previous_dual_lagrangian

        if return_multipliers:
            return self._primal_lagrangian, observed_multiplier_values
        else:
            return self._primal_lagrangian

    def backward(self) -> None:
        """
        Populates the gradient of the Lagrangian with respect to the primal and dual
        parameters.
        """

        # Populate the gradient of the Lagrangian w.r.t. the primal parameters
        if self._primal_lagrangian is not None and isinstance(self._primal_lagrangian, torch.Tensor):
            self._primal_lagrangian.backward()

        # Populate the gradient of the Lagrangian w.r.t. the dual parameters
        if self._dual_lagrangian is not None and isinstance(self._dual_lagrangian, torch.Tensor):
            self._dual_lagrangian.backward()


class ConstrainedMinimizationProblem(abc.ABC):
    """Base class for constrained minimization problems."""

    def __init__(self):
        self._state = CMPState()

    @property
    def state(self) -> CMPState:
        return self._state

    @state.setter
    def state(self, value: CMPState):
        self._state = value

    @abc.abstractmethod
    def closure(self, *args, **kwargs) -> CMPState:
        """
        Computes the state of the CMP based on the current value of the primal
        parameters.

        The signature of this abstract function may be changed to accommodate
        situations that require a model, (mini-batched) inputs/targets, or
        other arguments to be passed.

        Structuring the CMP class around this closure method, enables the re-use
        of shared sections of a computational graph. For example, consider a
        case where we want to minimize a model's cross entropy loss subject to
        a constraint on the entropy of its predictions. Both of these quantities
        depend on the predicted logits (on a minibatch). This closure-centric
        design allows flexible problem specifications while avoiding
        re-computation.
        """

    def defect_fn(self) -> CMPState:
        """
        Computes the constraints of the CMP based on the current value of the
        primal parameters. This function returns a
        :py:class:`cooper.problem.CMPState` collecting the values of the (proxy
        and/or non-proxy) constraints. Note that this returned ``CMPState`` may
        have a ``loss`` attribute with value ``None`` since, by design, the loss
        is not necessarily computed when evaluating `only` the constraints.

        The signature of this "abstract" function may be changed to accommodate
        situations that require a model, (mini-batched) inputs/targets, or
        other arguments to be passed.

        Depending on the problem at hand, the computation of the constraints
        can be compartimentalized in a way that is independent of the evaluation
        of the loss. Alternatively,
        :py:meth:`~.ConstrainedMinimizationProblem.defect_fn` may be used in the
        implementation of the :py:meth:`~.ConstrainedMinimizationProblem.closure`
        function.
        """
        raise NotImplementedError
