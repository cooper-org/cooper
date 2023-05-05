"""Cooper-related utilities for writing tests."""

from typing import Optional, Union

import pytest
import torch

import cooper


class Toy2dCMP(cooper.ConstrainedMinimizationProblem):
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
    Link to WolframAlpha query: https://tinyurl.com/ye8dw6t3
    """

    def __init__(self, use_ineq_constraints=False, use_constraint_surrogate=False, device=None):
        self.use_ineq_constraints = use_ineq_constraints
        self.use_constraint_surrogate = use_constraint_surrogate
        super().__init__()

        self.constraint_groups = []
        if self.use_ineq_constraints:
            multiplier_kwargs = {"shape": 1, "device": device}
            constraint_kwargs = {
                "constraint_type": cooper.ConstraintType.INEQUALITY,
                "formulation_type": cooper.FormulationType.LAGRANGIAN,
            }
            self.constraint_groups = [
                cooper.ConstraintGroup(**constraint_kwargs, multiplier_kwargs=multiplier_kwargs),
                cooper.ConstraintGroup(**constraint_kwargs, multiplier_kwargs=multiplier_kwargs),
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

        return loss_grad, torch.stack([cg0_grad, cg1_grad])

    def compute_violations(self, params) -> cooper.CMPState:
        """Evaluates the constraint violations for this CMP."""

        param_x, param_y = params() if callable(params) else params

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

        observed_constraints = [(self.constraint_groups[0], cg0_state), (self.constraint_groups[1], cg1_state)]

        return cooper.CMPState(loss=None, observed_constraints=observed_constraints)

    def compute_cmp_state(self, params):
        """Computes the state of the CMP at the current value of the primal parameters
        by evaluating the loss and constraints.
        """

        param_x, param_y = params() if callable(params) else params

        loss = param_x**2 + 2 * param_y**2
        cmp_state = cooper.CMPState(loss=loss)

        if self.use_ineq_constraints:
            violation_cmp_state = self.compute_violations(params=(param_x, param_y))
            cmp_state.observed_constraints = violation_cmp_state.observed_constraints

        return cmp_state


@pytest.fixture(params=[[0.0, -1.0], [0.1, 0.5]])
def Toy2dCMP_params_init(device, request):
    return torch.tensor(request.param, device=device)


@pytest.fixture(params=[True, False])
def Toy2dCMP_problem_properties(request, device):
    use_ineq_constraints = request.param
    if use_ineq_constraints:
        exact_solution = torch.tensor([2.0 / 3.0, 1.0 / 3.0], device=device)
    else:
        exact_solution = torch.tensor([0.0, 0.0], device=device)

    return dict(use_ineq_constraints=use_ineq_constraints, exact_solution=exact_solution)


@pytest.fixture(params=[True, False])
def use_multiple_primal_optimizers(request):
    return request.param


def build_params(use_multiple_primal_optimizers, params_init):
    if use_multiple_primal_optimizers:
        params = [torch.nn.Parameter(params_init[0]), torch.nn.Parameter(params_init[1])]
    else:
        params = torch.nn.Parameter(params_init)

    return params


def build_primal_optimizers(
    params, use_multiple_primal_optimizers, extrapolation=False, optimizer_names=None, optimizer_kwargs=None
):
    if use_multiple_primal_optimizers:

        if optimizer_names is None:
            optimizer_names = ["SGD", "Adam"] if not extrapolation else ["ExtraSGD", "ExtraAdam"]

        if optimizer_kwargs is None:
            optimizer_kwargs = [{"lr": 1e-2, "momentum": 0.3}, {"lr": 1e-2}]

        primal_optimizers = []
        for param, optimizer_name, kwargs in zip(params, optimizer_names, optimizer_kwargs):
            if not extrapolation:
                optimizer = getattr(torch.optim, optimizer_name)([param], **kwargs)
            else:
                optimizer = getattr(cooper.optim, optimizer_name)([param], **kwargs)

            primal_optimizers.append(optimizer)
    else:

        if optimizer_names is None:
            optimizer_names = "SGD" if not extrapolation else "ExtraSGD"
        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 1e-2}

        if not extrapolation:
            primal_optimizers = getattr(torch.optim, optimizer_names)([params], **optimizer_kwargs)
        else:
            primal_optimizers = getattr(cooper.optim, optimizer_names)([params], **optimizer_kwargs)

    return primal_optimizers


def build_params_and_primal_optimizers(
    use_multiple_primal_optimizers, params_init, extrapolation=False, optimizer_names=None, optimizer_kwargs=None
):
    params = build_params(use_multiple_primal_optimizers, params_init)
    primal_optimizers = build_primal_optimizers(
        params, use_multiple_primal_optimizers, extrapolation, optimizer_names, optimizer_kwargs
    )
    return params, primal_optimizers


def build_dual_optimizers(
    is_constrained,
    constraint_groups,
    extrapolation=False,
    dual_optimizer_name="SGD",
    dual_optimizer_kwargs={"lr": 1e-2},
):
    if is_constrained:
        dual_params = [{"params": constraint.multiplier.parameters()} for constraint in constraint_groups]
        if not extrapolation:
            dual_optimizers = getattr(torch.optim, dual_optimizer_name)(dual_params, **dual_optimizer_kwargs)
        else:
            dual_optimizers = getattr(cooper.optim, dual_optimizer_name)(dual_params, **dual_optimizer_kwargs)
    else:
        dual_optimizers = None

    return dual_optimizers


def build_cooper_optimizer_for_Toy2dCMP(
    primal_optimizers,
    constraint_groups,
    extrapolation=False,
    alternating=cooper.optim.AlternatingType.FALSE,
    dual_optimizer_name="SGD",
    dual_optimizer_kwargs={"lr": 1e-2},
) -> Union[cooper.optim.ConstrainedOptimizer, cooper.optim.UnconstrainedOptimizer]:

    is_constrained = len(constraint_groups) > 0
    dual_optimizers = build_dual_optimizers(
        is_constrained, constraint_groups, extrapolation, dual_optimizer_name, dual_optimizer_kwargs
    )

    cooper_optimizer = cooper.optim.utils.create_optimizer_from_kwargs(
        primal_optimizers=primal_optimizers,
        extrapolation=extrapolation,
        alternating=alternating,
        dual_optimizers=dual_optimizers,
        constraint_groups=constraint_groups,
    )

    return cooper_optimizer
