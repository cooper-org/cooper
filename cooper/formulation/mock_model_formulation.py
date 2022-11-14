from typing import Callable, no_type_check, Tuple, Optional, Union, List

from cooper.multipliers import MultiplierModel
from cooper.formulation.lagrangian_model import LagrangianModelFormulation
from cooper.problem import CMPState

import torch

class MockLagrangianModelFormulation(LagrangianModelFormulation):
    """
    Prototype of a new Lagrangian model formulation.
    """

    @no_type_check
    def backward(
        self,
        lagrangian: torch.Tensor,
        ignore_primal: bool = False,
        ignore_dual: bool = False,
    ):
        """
        Performs the actual backward computation which populates the gradients
        for the primal and dual variables.

        Args:
            lagrangian: Value of the computed Lagrangian based on which the
                gradients for the primal and dual variables are populated.
            ignore_primal: If ``True``, only the gradients with respect to the
                dual variables are populated (these correspond to the constraint
                violations). This feature is mainly used in conjunction with
                ``alternating`` updates, which require updating the multipliers
                based on the constraints violation *after* having updated the
                primal parameters. Defaults to False.
            ignore_dual: If ``True``, the gradients with respect to the dual
                variables are not populated.
        """

        if ignore_primal:
            # Only compute gradients wrt Lagrange multipliers
            # No need to call backward on Lagrangian as the dual variables have
            # been detached when computing the `weighted_violation`s
            pass
        else:
            # Compute gradients wrt _primal_ parameters only.
            # The gradient for the dual variables is computed based on the
            # non-proxy violations below.
            lagrangian.backward()

        # Fill in the gradients for the dual variables based on the violation of
        # the non-proxy constraints
        if not ignore_dual:
            dual_vars = self.dual_parameters
            new_loss = lagrangian - self.accumulated_violation_dot_prod
            new_loss.backward(inputs=dual_vars)
