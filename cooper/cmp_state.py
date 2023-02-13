from typing import Iterable, List, Optional, Tuple

import torch

from cooper.constraints import ConstraintGroup, ConstraintState


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
        observed_constraints: Iterable[Tuple[ConstraintGroup, Optional[ConstraintState]]] = None,
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

            constraint_group = constraint_tuple[0]
            constraint_state = constraint_tuple[1] if len(constraint_tuple) == 2 else constraint_group.state

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
