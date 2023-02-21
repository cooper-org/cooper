"""Cooper-related utilities for writing tests."""

import functools
from typing import List, Type

import torch

import cooper


class Toy2dCMP:
    """
    Simple test on a 2D quadratically-constrained quadratic programming problem
        min x**2 + 2*y**2
        st.
            x + y >= 1
            x**2 + y <= 1

    If hard violations are used, the "differentiable" surrogates are set to:
            0.9 * x + y >= 1
            x**2 + 0.9 * y <= 1

    This is a convex optimization problem.

    The constraint levels of the differentiable surrogates are not strictly
    required since these functions are only employed via their gradients, thus
    the constant contribution of the constraint level disappears. We include
    them here for readability.

    Verified solution from WolframAlpha of the original constrained problem:
        (x=2/3, y=1/3)
    Link to WolframAlpha query: https://tinyurl.com/ye8dw6t3
    """

    def __init__(self, use_ineq_constraints=False, use_constraint_surrogate=False):
        self.use_ineq_constraints = use_ineq_constraints
        self.use_constraint_surrogate = use_constraint_surrogate
        super().__init__()

        if self.use_ineq_constraints:
            self.cg1 = cooper.ConstraintGroup(constraint_type="ineq", formulation_type="lagrangian", shape=1)
            self.cg2 = cooper.ConstraintGroup(constraint_type="ineq", formulation_type="lagrangian", shape=1)

    def analytical_gradients(self, params):
        """Returns the analytical gradients of the loss and constraints for a given
        value of the parameters."""

        param_x, param_y = params() if callable(params) else params

        # Params are detached and cloned for safety
        param_x = param_x.detach().clone()
        param_y = param_y.detach().clone()

        loss_grad = torch.stack([2 * param_x, 4 * param_y])

        if not self.use_ineq_constraints:
            return loss_grad

        if not self.use_constraint_surrogate:
            cg1_grad = torch.tensor([-1.0, -1.0], device=param_x.device)
            cg2_grad = torch.tensor([2 * param_x, 1.0], device=param_x.device)
        else:
            cg1_grad = torch.tensor([-0.9, -1.0], device=param_x.device)
            cg2_grad = torch.tensor([2 * param_x, 0.9], device=param_x.device)

        return loss_grad, cg1_grad, cg2_grad

    def closure(self, params):

        param_x, param_y = params() if callable(params) else params

        loss = param_x**2 + 2 * param_y**2

        if self.use_ineq_constraints:

            # Orig constraint: x + y \ge 1
            cg1_violation = -param_x - param_y + 1.0
            cg1_surrogate = -0.9 * param_x - param_y + 1.0

            # Orig constraint: x**2 + y \le 1.0
            cg2_violation = param_x**2 + param_y - 1.0
            cg2_surrogate = param_x**2 + 0.9 * param_y - 1.0

            if self.use_constraint_surrogate:
                self.cg1.state = cooper.ConstraintState(violation=cg1_surrogate, strict_violation=cg1_violation)
                self.cg2.state = cooper.ConstraintState(violation=cg2_surrogate, strict_violation=cg2_violation)
            else:
                self.cg1.state = cooper.ConstraintState(violation=cg1_violation)
                self.cg2.state = cooper.ConstraintState(violation=cg1_violation)

            observed_constraints = [self.cg1, self.cg2]
        else:
            observed_constraints = []

        return cooper.CMPState(loss=loss, observed_constraints=observed_constraints)


def get_optimizer_from_str(optimizer_str: str) -> Type[torch.optim.Optimizer]:
    """
    Returns an optimizer class from the string name of the optimizer.
    """
    # TODO(juan43ramirez): this helper function could be useful for Cooper users.
    # Consider moving it to the optim module.
    try:
        return getattr(cooper.optim, optimizer_str)
    except:
        return getattr(torch.optim, optimizer_str)


def build_params_from_init(init: List[float], device: torch.device) -> torch.nn.Parameter:
    """Builds a torch.nn.Parameter object from a list of initial values."""
    return [torch.nn.Parameter(torch.tensor([elem], device=device, requires_grad=True)) for elem in init]


def mktensor(device):
    return functools.partial(torch.tensor, device=device)
