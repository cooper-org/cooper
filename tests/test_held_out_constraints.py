import cooper_test_utils
import pytest
import testing_utils
import torch

import cooper


class RandomConstraintsToy2dCMP(cooper.ConstrainedMinimizationProblem):
    """
    Simple test on a 2D quadratic programming problem with quadratic constraints
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

    """

    def __init__(self, use_constraint_surrogate=False, device=None, observe_probability=1.0):
        self.use_constraint_surrogate = use_constraint_surrogate
        self.observe_probability = observe_probability

        multiplier_kwargs = {"shape": 2, "device": device, "is_indexed": True}
        constraint_kwargs = {
            "constraint_type": cooper.ConstraintType.INEQUALITY,
            "formulation_type": cooper.FormulationType.LAGRANGIAN,
        }

        self.constraint_group = cooper.ConstraintGroup(**constraint_kwargs, multiplier_kwargs=multiplier_kwargs)

    def analytical_gradients(self, params):
        """Returns the analytical gradients of the loss and constraints for a given
        value of the parameters."""

        param_x, param_y = params() if callable(params) else params

        # Params are detached and cloned for safety
        param_x = param_x.detach().clone()
        param_y = param_y.detach().clone()

        # Gradient of x^2 + 2 * y^2 is [2 * x, 4 * y]
        loss_grad = torch.stack([2 * param_x, 4 * param_y])

        if self.use_constraint_surrogate:
            # The constraint surrogates take precedence over the `strict_violation`
            # when computing the gradient of the Lagrangian wrt the primal parameters.
            cg0_grad = torch.tensor([-0.9, -1.0], device=param_x.device)
            cg1_grad = torch.tensor([2 * param_x, 0.9], device=param_x.device)
        else:
            cg0_grad = torch.tensor([-1.0, -1.0], device=param_x.device)
            cg1_grad = torch.tensor([2 * param_x, 1.0], device=param_x.device)

        return loss_grad, torch.stack([cg0_grad, cg1_grad])

    def compute_cmp_state(self, params):
        """Computes the state of the CMP at the current value of the primal parameters
        by evaluating the loss and constraints.
        """

        param_x, param_y = params() if callable(params) else params

        loss = param_x**2 + 2 * param_y**2

        cg0_violation = -param_x - param_y + 1.0
        cg1_violation = param_x**2 + param_y - 1.0

        if self.use_constraint_surrogate:
            # The constraint surrogates take precedence over the `strict_violation`
            # when computing the gradient of the Lagrangian wrt the primal variables

            # Orig constraint: x + y \ge 1
            cg0_surrogate = -0.9 * param_x - param_y + 1.0

            # Orig constraint: x**2 + y \le 1.0
            cg1_surrogate = param_x**2 + 0.9 * param_y - 1.0

            violation = torch.stack([cg0_surrogate, cg1_surrogate])
            strict_violation = torch.stack([cg0_violation, cg1_violation])
        else:
            violation = torch.stack([cg0_violation, cg1_violation])
            strict_violation = violation

        # Randomly keep or drop each entry in violation and strict_violation
        constraint_features = (torch.rand_like(violation) < self.observe_probability).nonzero().flatten()
        strict_constraint_features = (torch.rand_like(strict_violation) < self.observe_probability).nonzero().flatten()

        violation = violation[constraint_features]
        strict_violation = strict_violation[strict_constraint_features]

        constraint_state = cooper.ConstraintState(
            violation=violation,
            strict_violation=strict_violation,
            constraint_features=constraint_features,
            strict_constraint_features=strict_constraint_features,
        )

        return cooper.CMPState(loss=loss, observed_constraints=[(self.constraint_group, constraint_state)])


@pytest.fixture(params=[0.9])
def observe_probability(request):
    return request.param


# def test_convergence(
#     Toy2dCMP_problem_properties, Toy2dCMP_params_init, use_multiple_primal_optimizers, device, observe_probability
# ):
#     """Tests the correct instantiation of a simple CMP class for a 2-dimensional
#     convex (quadratic) constrained problem.

#     Verifies that executing simultaneous updates on this problem converges to the
#     analytical solution to this problem. This check is also performed for the case where
#     the CMP has no constraints -- note that the solution to the unconstrained problem is
#     different.
#     """

#     params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
#         use_multiple_primal_optimizers, Toy2dCMP_params_init
#     )

#     use_ineq_constraints = Toy2dCMP_problem_properties["use_ineq_constraints"]
#     if not use_ineq_constraints:
#         pytest.skip("This test is only relevant for constrained problems.")

#     use_constraint_surrogate = False

#     cmp = RandomConstraintsToy2dCMP(
#         device=device, use_constraint_surrogate=use_constraint_surrogate, observe_probability=observe_probability
#     )

#     cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(primal_optimizers, [cmp.constraint_group])

#     for step_id in range(1500):
#         compute_cmp_state_fn = lambda: cmp.compute_cmp_state(params)
#         cmp_state, lagrangian_store = cooper_optimizer.roll(compute_cmp_state_fn=compute_cmp_state_fn)

#     for param, exact_solution in zip(params, Toy2dCMP_problem_properties["exact_solution"]):
#         assert torch.allclose(param, exact_solution)
