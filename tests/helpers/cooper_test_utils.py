"""Cooper-related utilities for writing tests."""

import pytest
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

    The constraint levels of the differentiable surrogates are not strictly required
    since these functions are only employed via their gradients, thus the constant
    contribution of the constraint level disappears. We include them here for
    readability.

    Verified solution of the original constrained problem:
        (x=2/3, y=1/3)
    Link to WolframAlpha query: https://tinyurl.com/ye8dw6t3
    """

    def __init__(self, use_ineq_constraints=False, use_constraint_surrogate=False, device=None):
        self.use_ineq_constraints = use_ineq_constraints
        self.use_constraint_surrogate = use_constraint_surrogate
        super().__init__()

        self.constraint_groups = []
        if self.use_ineq_constraints:
            self.constraint_groups = [
                cooper.ConstraintGroup(constraint_type="ineq", formulation_type="lagrangian", shape=1, device=device),
                cooper.ConstraintGroup(constraint_type="ineq", formulation_type="lagrangian", shape=1, device=device),
            ]

    def analytical_gradients(self, params):
        """Returns the analytical gradients of the loss and constraints for a given
        value of the parameters."""

        param_x, param_y = params() if callable(params) else params

        # Params are detached and cloned for safety
        param_x = param_x.detach().clone()
        param_y = param_y.detach().clone()

        # Gradient of x^2 + 2 * y^2 is [2 * x, 4 * y]
        loss_grad = torch.stack([2 * param_x, 4 * param_y])

        if not self.use_ineq_constraints:
            return loss_grad

        if self.use_constraint_surrogate:
            # The constraint surrogates take precedence over the `strict_violation`
            # when computing the gradient of the Lagrangian wrt the primal parameters.
            cg0_grad = torch.tensor([-0.9, -1.0], device=param_x.device)
            cg1_grad = torch.tensor([2 * param_x, 0.9], device=param_x.device)
        else:
            cg0_grad = torch.tensor([-1.0, -1.0], device=param_x.device)
            cg1_grad = torch.tensor([2 * param_x, 1.0], device=param_x.device)

        return loss_grad, cg0_grad, cg1_grad

    def compute_cmp_state(self, params):
        """
        Computes the state of the CMP at the current value of the primal parameters by
        evaluating the loss and constraints.
        """

        param_x, param_y = params() if callable(params) else params

        loss = param_x**2 + 2 * param_y**2

        if self.use_ineq_constraints:

            cg0_violation = -param_x - param_y + 1.0
            cg1_violation = param_x**2 + param_y - 1.0

            if self.use_constraint_surrogate:
                # The constraint surrogates take precedence over the `strict_violation`
                # when computing the gradient of the Lagrangian wrt the primal variables

                # Orig constraint: x + y \ge 1
                cg0_surrogate = -0.9 * param_x - param_y + 1.0
                cg0_state = cooper.ConstraintState(violation=cg0_surrogate, strict_violation=cg0_violation)

                # Orig constraint: x**2 + y \le 1.0
                cg1_surrogate = param_x**2 + 0.9 * param_y - 1.0
                cg1_state = cooper.ConstraintState(violation=cg1_surrogate, strict_violation=cg1_violation)
            else:
                cg0_state = cooper.ConstraintState(violation=cg0_violation)
                cg1_state = cooper.ConstraintState(violation=cg1_violation)

            self.constraint_groups[0].state = cg0_state
            self.constraint_groups[1].state = cg1_state

            observed_constraints = [self.constraint_groups[0], self.constraint_groups[1]]
        else:
            observed_constraints = []

        return cooper.CMPState(loss=loss, observed_constraints=observed_constraints)


@pytest.fixture(params=[[0.0, -1.0], [0.1, 0.5]])
def Toy2dCMP_params_init(device, request):
    return torch.tensor(request.param, device=device)


@pytest.fixture(params=[True, False])
def Toy2dCMP_problem_properties(request, device):
    use_ineq_constraints = request.param
    if use_ineq_constraints:
        solution = torch.tensor([2.0 / 3.0, 1.0 / 3.0], device=device)
    else:
        solution = torch.tensor([0.0, 0.0], device=device)

    return dict(use_ineq_constraints=use_ineq_constraints, solution=solution)
