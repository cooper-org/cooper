"""Cooper-related utilities for writing tests."""

from copy import deepcopy
from enum import Enum
from typing import Optional, Type

import pytest
import torch

import cooper
from cooper.optim import CooperOptimizer, UnconstrainedOptimizer, constrained_optimizers
from cooper.utils import OneOrSequence


class SquaredNormLinearCMP(cooper.ConstrainedMinimizationProblem):
    """
    Minimization problem for minimizing the square norm of a vector under linear constraints:
        min ||x||^2
        st. Ax <= b
        &   Cx == d

    This is a convex optimization problem with linear inequality constraints.

    Parameters:
        A (torch.Tensor): Coefficient matrix for the linear inequality constraints.
        b (torch.Tensor): Bias vector for the linear inequality constraints.
        device (torch.device): The device tensors will be allocated on.
    """

    def __init__(
        self,
        has_ineq_constraint: bool = True,
        has_eq_constraint: bool = True,
        ineq_use_surrogate: bool = False,
        eq_use_surrogate: bool = False,
        A: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        C: Optional[torch.Tensor] = None,
        d: Optional[torch.Tensor] = None,
        ineq_formulation_type: Type[cooper.Formulation] = cooper.LagrangianFormulation,
        ineq_multiplier_type: Type[cooper.multipliers.Multiplier] = cooper.multipliers.DenseMultiplier,
        ineq_penalty_coefficient_type: Optional[Type[cooper.multipliers.PenaltyCoefficient]] = None,
        eq_formulation_type: Type[cooper.Formulation] = cooper.LagrangianFormulation,
        eq_multiplier_type: Type[cooper.multipliers.Multiplier] = cooper.multipliers.DenseMultiplier,
        eq_penalty_coefficient_type: Optional[Type[cooper.multipliers.PenaltyCoefficient]] = None,
        device="cpu",
    ):
        super().__init__()
        self.has_ineq_constraint = has_ineq_constraint
        self.has_eq_constraint = has_eq_constraint
        self.ineq_use_surrogate = ineq_use_surrogate
        self.eq_use_surrogate = eq_use_surrogate
        self.ineq_multiplier_type = ineq_multiplier_type
        self.eq_multiplier_type = eq_multiplier_type
        self.A = A.to(device) if A is not None else None
        self.b = b.to(device) if b is not None else None
        self.C = C.to(device) if C is not None else None
        self.d = d.to(device) if d is not None else None
        self.device = device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(0)

        if has_ineq_constraint:
            ineq_penalty_coefficient = None
            if ineq_penalty_coefficient_type is not None:
                ineq_penalty_coefficient = ineq_penalty_coefficient_type(init=torch.ones(b.numel(), device=device))

            self.ineq_constraints = cooper.Constraint(
                constraint_type=cooper.ConstraintType.INEQUALITY,
                formulation_type=ineq_formulation_type,
                multiplier=ineq_multiplier_type(
                    constraint_type=cooper.ConstraintType.INEQUALITY, num_constraints=b.numel(), device=device
                ),
                penalty_coefficient=ineq_penalty_coefficient,
            )

        if has_eq_constraint:
            eq_penalty_coefficient = None
            if eq_penalty_coefficient_type is not None:
                eq_penalty_coefficient = eq_penalty_coefficient_type(init=torch.ones(d.numel(), device=device))

            self.eq_constraints = cooper.Constraint(
                constraint_type=cooper.ConstraintType.EQUALITY,
                formulation_type=eq_formulation_type,
                multiplier=eq_multiplier_type(
                    constraint_type=cooper.ConstraintType.EQUALITY, num_constraints=d.numel(), device=device
                ),
                penalty_coefficient=eq_penalty_coefficient,
            )

    def _compute_eq_or_ineq_violations(self, x, A_or_C, b_or_d, has_constraint, use_surrogate, multiplier_type):
        if not has_constraint:
            return None

        violation = A_or_C @ x - b_or_d

        constraint_features = None
        if multiplier_type == cooper.multipliers.IndexedMultiplier:
            constraint_features = torch.randperm(b_or_d.numel(), generator=self.generator, device=self.device)
            # we assume 80% of the constraints are observed
            constraint_features = constraint_features[: 4 * b_or_d.numel() // 5]
            violation = violation[constraint_features]

        strict_violation = None
        strict_constraint_features = None
        if use_surrogate:
            strict_violation = torch.mm(
                A_or_C + 0.1 * torch.randn(A_or_C.shape, generator=self.generator, device=self.device), x
            )
            if multiplier_type == cooper.multipliers.IndexedMultiplier:
                strict_constraint_features = torch.randperm(
                    b_or_d.numel(), generator=self.generator, device=self.device
                )
                # we assume 80% of the constraints are observed
                strict_constraint_features = strict_constraint_features[: 4 * b_or_d.numel() // 5]
                strict_violation = strict_violation[strict_constraint_features]

        return cooper.ConstraintState(
            violation=violation,
            constraint_features=constraint_features,
            strict_violation=strict_violation,
            strict_constraint_features=strict_constraint_features,
        )

    def compute_violations(self, x: torch.Tensor) -> cooper.CMPState:
        """
        Computes the constraint violations for the given parameters.
        """
        ineq_state = self._compute_eq_or_ineq_violations(
            x, self.A, self.b, self.has_ineq_constraint, self.ineq_use_surrogate, self.ineq_multiplier_type
        )
        eq_state = self._compute_eq_or_ineq_violations(
            x, self.C, self.d, self.has_eq_constraint, self.eq_use_surrogate, self.eq_multiplier_type
        )

        observed_constraints = {}
        if self.has_ineq_constraint:
            observed_constraints[self.ineq_constraints] = ineq_state
        if self.has_eq_constraint:
            observed_constraints[self.eq_constraints] = eq_state

        return cooper.CMPState(observed_constraints=observed_constraints)

    def compute_cmp_state(self, x: torch.Tensor) -> cooper.CMPState:
        """
        Computes the state of the CMP at the current value of the primal parameters
        by evaluating the loss and constraints.
        """
        loss = torch.sum(x**2)
        violation_state = self.compute_violations(x)
        cmp_state = cooper.CMPState(loss=loss, observed_constraints=violation_state.observed_constraints)
        return cmp_state


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

    For a value of C=1, the solution is:
        (x=1/2, y=1/4, s0=0, s1=1/4)

    Link to WolframAlpha queries:
        Standard CMP: https://tinyurl.com/ye8dw6t3
    """

    def __init__(
        self,
        use_ineq_constraints=False,
        use_constraint_surrogate=False,
        constraint_type: cooper.ConstraintType = cooper.ConstraintType.INEQUALITY,
        formulation_type: Type[cooper.Formulation] = cooper.LagrangianFormulation,
        penalty_coefficients: Optional[tuple[cooper.multipliers.PenaltyCoefficient, ...]] = None,
        device=None,
    ):
        super().__init__()

        self.use_ineq_constraints = use_ineq_constraints
        self.use_constraint_surrogate = use_constraint_surrogate

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
        """Evaluates only the constraint violations for this CMP."""
        if params is None:
            raise NotImplementedError()

        param_x, param_y = params() if callable(params) else params

        cg0_violation = -param_x - param_y + 1.0
        cg1_violation = param_x**2 + param_y - 1.0

        if self.use_constraint_surrogate:
            # The constraint surrogates take precedence over the `strict_violation`
            # when computing the gradient of the Lagrangian wrt the primal variables

            # Orig constraint: x + y \ge 1
            cg0_surrogate = -0.9 * param_x - param_y + 1.0

            # Orig constraint: x**2 + y \le 1.0
            cg1_surrogate = param_x**2 + 0.9 * param_y - 1.0

            cg0_state = cooper.ConstraintState(violation=cg0_surrogate, strict_violation=cg0_violation)
            cg1_state = cooper.ConstraintState(violation=cg1_surrogate, strict_violation=cg1_violation)
        else:
            cg0_state = cooper.ConstraintState(violation=cg0_violation)
            cg1_state = cooper.ConstraintState(violation=cg1_violation)

        observed_constraints = {self.constraint_0: cg0_state, self.constraint_1: cg1_state}

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
    cmp_properties = dict(use_ineq_constraints=use_ineq_constraints)

    if use_ineq_constraints:
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
        params = [torch.nn.Parameter(params_init[0].clone()), torch.nn.Parameter(params_init[1].clone())]
    else:
        params = torch.nn.Parameter(params_init.clone())

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
    dual_parameters,
    augmented_lagrangian=False,
    dual_optimizer_class=torch.optim.SGD,
    dual_optimizer_kwargs={"lr": 1e-2},
):

    # Make copy of this fixture since we are modifying in-place
    dual_optimizer_kwargs = deepcopy(dual_optimizer_kwargs)
    dual_optimizer_kwargs["maximize"] = True

    if augmented_lagrangian:
        assert dual_optimizer_class == torch.optim.SGD
        dual_optimizer_kwargs["lr"] = 1.0

    if dual_optimizer_class == torch.optim.SGD:
        # SGD does not support `foreach=True` (the default for 2.0.0) when the
        # parameters use sparse gradients. Disabling foreach.
        dual_optimizer_kwargs["foreach"] = False

    return dual_optimizer_class(dual_parameters, **dual_optimizer_kwargs)


def create_optimizer_from_kwargs(
    cooper_optimizer_class: Type[CooperOptimizer],
    cmp: cooper.ConstrainedMinimizationProblem,
    primal_optimizers: OneOrSequence[torch.optim.Optimizer],
    dual_optimizers: Optional[OneOrSequence[torch.optim.Optimizer]] = None,
) -> CooperOptimizer:
    """Creates a constrained or unconstrained optimizer from a set of keyword arguments."""

    if dual_optimizers is None:
        if cooper_optimizer_class != UnconstrainedOptimizer:
            raise ValueError("Dual optimizers must be provided for constrained optimization problems.")
        optimizer_kwargs = dict(primal_optimizers=primal_optimizers, cmp=cmp)
    else:
        optimizer_kwargs = dict(primal_optimizers=primal_optimizers, dual_optimizers=dual_optimizers, cmp=cmp)

    return cooper_optimizer_class(**optimizer_kwargs)


class AlternationType(Enum):
    FALSE = False
    PRIMAL_DUAL = "PrimalDual"
    DUAL_PRIMAL = "DualPrimal"


def build_cooper_optimizer_for_Toy2dCMP(
    cmp,
    primal_optimizers,
    extrapolation: bool = False,
    augmented_lagrangian: bool = False,
    alternation_type: AlternationType = AlternationType.FALSE,
    dual_optimizer_class=torch.optim.SGD,
    dual_optimizer_kwargs={"lr": 1e-2},
) -> CooperOptimizer:

    dual_optimizers = None
    if len(list(cmp.constraints())) != 0:
        dual_optimizers = build_dual_optimizers(
            dual_parameters=cmp.dual_parameters(),
            augmented_lagrangian=augmented_lagrangian,
            dual_optimizer_class=dual_optimizer_class,
            dual_optimizer_kwargs=dual_optimizer_kwargs,
        )

    if dual_optimizers is None:
        cooper_optimizer_class = cooper.optim.UnconstrainedOptimizer
    else:
        if extrapolation:
            cooper_optimizer_class = constrained_optimizers.ExtrapolationConstrainedOptimizer
        else:
            if alternation_type == AlternationType.DUAL_PRIMAL:
                cooper_optimizer_class = constrained_optimizers.AlternatingDualPrimalOptimizer
            elif alternation_type == AlternationType.PRIMAL_DUAL:
                cooper_optimizer_class = constrained_optimizers.AlternatingPrimalDualOptimizer
            else:
                cooper_optimizer_class = constrained_optimizers.SimultaneousOptimizer

    cooper_optimizer = create_optimizer_from_kwargs(
        cooper_optimizer_class=cooper_optimizer_class,
        cmp=cmp,
        primal_optimizers=primal_optimizers,
        dual_optimizers=dual_optimizers,
    )

    return cooper_optimizer
