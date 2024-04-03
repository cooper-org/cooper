"""Cooper-related utilities for writing tests."""

import itertools
from copy import deepcopy
from typing import Optional, Type, Union

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

    When slack variables are enabled, the problem is reformulated as:
        min x**2 + 2*y**2 + C * (s0**2 + s1**2)
        st.
            x + y >= 1 - s0
            x**2 + y <= 1 + s1
            s0, s1 >= 0

    For a value of C=1, the solution is:
        (x=1/2, y=1/4, s0=0, s1=1/4)

    Link to WolframAlpha queries:
        Standard CMP: https://tinyurl.com/ye8dw6t3
        CMP with slack variables: https://tinyurl.com/bds5b3yj
    """

    def __init__(
        self,
        use_ineq_constraints=False,
        use_constraint_surrogate=False,
        constraint_type: cooper.ConstraintType = cooper.ConstraintType.INEQUALITY,
        formulation_type: Type[cooper.Formulation] = cooper.LagrangianFormulation,
        slack_variables: Optional[tuple[cooper.constraints.SlackVariable]] = None,
        penalty_coefficients: Optional[tuple[cooper.multipliers.PenaltyCoefficient]] = None,
        device=None,
    ):
        super().__init__()

        self.use_ineq_constraints = use_ineq_constraints
        self.use_constraint_surrogate = use_constraint_surrogate
        self.slack_variables = slack_variables

        if self.use_ineq_constraints:
            for ix in range(2):
                multiplier = cooper.multipliers.DenseMultiplier(
                    constraint_type=constraint_type, num_constraints=1, device=device
                )
                penalty_coefficient = penalty_coefficients[ix] if penalty_coefficients is not None else None

                constraint = cooper.Constraint(
                    constraint_type=constraint_type,
                    formulation_type=formulation_type,
                    multiplier=multiplier,
                    penalty_coefficient=penalty_coefficient,
                )
                setattr(self, f"constraint_{ix}", constraint)

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

        if self.slack_variables is not None:
            cg0_violation -= self.slack_variables[0]()
            cg1_violation -= self.slack_variables[1]()

        if self.use_constraint_surrogate:
            # The constraint surrogates take precedence over the `strict_violation`
            # when computing the gradient of the Lagrangian wrt the primal variables

            # Orig constraint: x + y \ge 1
            cg0_surrogate = -0.9 * param_x - param_y + 1.0

            # Orig constraint: x**2 + y \le 1.0
            cg1_surrogate = param_x**2 + 0.9 * param_y - 1.0

            if self.slack_variables is not None:
                cg0_surrogate -= self.slack_variables[0]()
                cg1_surrogate -= self.slack_variables[1]()

            cg0_state = cooper.ConstraintState(violation=cg0_surrogate, strict_violation=cg0_violation)
            cg1_state = cooper.ConstraintState(violation=cg1_surrogate, strict_violation=cg1_violation)
        else:
            cg0_state = cooper.ConstraintState(violation=cg0_violation)
            cg1_state = cooper.ConstraintState(violation=cg1_violation)

        observed_constraints = [(getattr(self, "constraint_0"), cg0_state), (getattr(self, "constraint_1"), cg1_state)]

        return cooper.CMPState(loss=None, observed_constraints=observed_constraints)

    def compute_cmp_state(self, params):
        """Computes the state of the CMP at the current value of the primal parameters
        by evaluating the loss and constraints.
        """

        param_x, param_y = params() if callable(params) else params

        loss = param_x**2 + 2 * param_y**2

        if self.slack_variables is not None:
            loss += self.slack_variables[0]() ** 2 + self.slack_variables[1]() ** 2

        cmp_state = cooper.CMPState(loss=loss)

        if self.use_ineq_constraints:
            violation_cmp_state = self.compute_violations(params=(param_x, param_y))
            cmp_state.observed_constraints = violation_cmp_state.observed_constraints

        return cmp_state


@pytest.fixture(params=[[0.0, -1.0], [0.1, 0.5]])
def Toy2dCMP_params_init(device, request):
    return torch.tensor(request.param, device=device)


@pytest.fixture(params=list(itertools.product([True, False], repeat=2)))
def Toy2dCMP_problem_properties(request, device):
    use_ineq_constraints, use_slack_variables = request.param

    use_slack_variables = False
    cmp_properties = dict(use_ineq_constraints=use_ineq_constraints, use_slack_variables=use_slack_variables)
    if use_ineq_constraints:
        if use_slack_variables:
            exact_solution = torch.tensor([1.0 / 2.0, 1.0 / 4.0], device=device)
            cmp_properties["slack_variables"] = torch.tensor([0.0, 1.0 / 4.0])
        else:
            exact_solution = torch.tensor([2.0 / 3.0, 1.0 / 3.0], device=device)
    else:
        exact_solution = torch.tensor([0.0, 0.0], device=device)

    cmp_properties["exact_solution"] = exact_solution

    return cmp_properties


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
    multipliers,
    extrapolation=False,
    augmented_lagrangian=False,
    dual_optimizer_class=torch.optim.SGD,
    dual_optimizer_kwargs={"lr": 1e-2},
):

    # Make copy of this fixture since we are modifying in-place
    dual_optimizer_kwargs = deepcopy(dual_optimizer_kwargs)
    dual_optimizer_kwargs["maximize"] = True

    dual_params = [{"params": _.parameters()} for _ in multipliers]
    if extrapolation:
        # Extrapolation optimizers are part of the `cooper.optim` module
        dual_optimizers = dual_optimizer_class(dual_params, **dual_optimizer_kwargs)
    else:
        if augmented_lagrangian:
            assert dual_optimizer_class == torch.optim.SGD
            dual_optimizer_kwargs["lr"] = 1.0

        if dual_optimizer_class == torch.optim.SGD:
            # SGD does not support `foreach=True` (the default for 2.0.0) when the
            # parameters use sparse gradients. Disabling foreach.
            dual_optimizer_kwargs["foreach"] = False

        dual_optimizers = dual_optimizer_class(dual_params, **dual_optimizer_kwargs)

    return dual_optimizers


def build_cooper_optimizer_for_Toy2dCMP(
    primal_optimizers,
    cmp,
    extrapolation: bool = False,
    augmented_lagrangian: bool = False,
    alternation_type: cooper.optim.AlternationType = cooper.optim.AlternationType.FALSE,
    dual_optimizer_class=torch.optim.SGD,
    dual_optimizer_kwargs={"lr": 1e-2},
) -> Union[cooper.optim.ConstrainedOptimizer, cooper.optim.UnconstrainedOptimizer]:

    dual_optimizers = None
    if len(list(cmp.constraints())) != 0:
        dual_optimizers = build_dual_optimizers(
            multipliers=cmp.multipliers(),
            augmented_lagrangian=augmented_lagrangian,
            extrapolation=extrapolation,
            dual_optimizer_class=dual_optimizer_class,
            dual_optimizer_kwargs=dual_optimizer_kwargs,
        )

    cooper_optimizer = cooper.optim.utils.create_optimizer_from_kwargs(
        primal_optimizers=primal_optimizers,
        cmp=cmp,
        dual_optimizers=dual_optimizers,
        extrapolation=extrapolation,
        alternation_type=alternation_type,
        augmented_lagrangian=augmented_lagrangian,
    )

    return cooper_optimizer
