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


def test_cmp():

    params = torch.nn.Parameter(torch.tensor([0.1, 0.5]))
    cmp = Toy2dCMP(use_ineq_constraints=True)

    cmp_state = cmp.closure(params)

    lagrangian, multipliers = cmp_state.populate_lagrangian(return_multipliers=True)
    cmp_state.backward()

    breakpoint()
