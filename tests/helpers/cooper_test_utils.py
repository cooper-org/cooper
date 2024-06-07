"""Cooper-related utilities for writing tests."""

from copy import deepcopy
from enum import Enum
from typing import Iterable, Optional, Type

import cvxpy as cp
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
        num_variables: int,
        has_ineq_constraint: bool = False,
        has_eq_constraint: bool = False,
        ineq_use_surrogate: bool = False,
        eq_use_surrogate: bool = False,
        A: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        C: Optional[torch.Tensor] = None,
        d: Optional[torch.Tensor] = None,
        ineq_formulation_type: Type[cooper.Formulation] = cooper.LagrangianFormulation,
        ineq_multiplier_type: Type[cooper.multipliers.Multiplier] = cooper.multipliers.DenseMultiplier,
        ineq_penalty_coefficient_type: Optional[Type[cooper.multipliers.PenaltyCoefficient]] = None,
        ineq_observed_constraint_ratio: float = 1.0,
        eq_formulation_type: Type[cooper.Formulation] = cooper.LagrangianFormulation,
        eq_multiplier_type: Type[cooper.multipliers.Multiplier] = cooper.multipliers.DenseMultiplier,
        eq_penalty_coefficient_type: Optional[Type[cooper.multipliers.PenaltyCoefficient]] = None,
        eq_observed_constraint_ratio: float = 1.0,
        device="cpu",
    ):
        super().__init__()
        self.num_variables = num_variables
        self.has_ineq_constraint = has_ineq_constraint
        self.has_eq_constraint = has_eq_constraint
        self.ineq_use_surrogate = ineq_use_surrogate
        self.eq_use_surrogate = eq_use_surrogate
        self.ineq_multiplier_type = ineq_multiplier_type
        self.eq_multiplier_type = eq_multiplier_type
        self.ineq_observed_constraint_ratio = ineq_observed_constraint_ratio
        self.eq_observed_constraint_ratio = eq_observed_constraint_ratio
        self.A = A.to(device) if A is not None else None
        self.b = b.to(device) if b is not None else None
        self.C = C.to(device) if C is not None else None
        self.d = d.to(device) if d is not None else None
        self.device = device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(0)

        self.A_sur = None
        if has_ineq_constraint and ineq_use_surrogate:
            # Make noise generator deterministic, so that we don't deal with stochastic constraints
            noise = 1e-1 * torch.randn(A.shape, generator=self.generator, device=self.device)
            self.A_sur = A + noise

        self.C_sur = None
        if has_eq_constraint and eq_use_surrogate:
            # Make noise generator deterministic, so that we don't deal with stochastic constraints
            noise = 1e-1 * torch.randn(C.shape, generator=self.generator, device=self.device)
            self.C_sur = C + noise

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

    def _compute_violations(self, x, lhs, rhs, lhs_sur, multiplier_type, observed_constraint_ratio):

        violation = torch.matmul(lhs, x) - rhs

        constraint_features = None
        if multiplier_type == cooper.multipliers.IndexedMultiplier:
            constraint_features = torch.randperm(rhs.numel(), generator=self.generator, device=self.device)
            constraint_features = constraint_features[: int(observed_constraint_ratio * rhs.numel())]
            violation = violation[constraint_features]

        strict_violation = None
        strict_constraint_features = None
        if lhs_sur is not None:
            strict_violation = violation.clone()
            violation = torch.matmul(lhs_sur, x) - rhs

            if multiplier_type == cooper.multipliers.IndexedMultiplier:
                strict_constraint_features = torch.randperm(rhs.numel(), generator=self.generator, device=self.device)
                strict_constraint_features = strict_constraint_features[: int(observed_constraint_ratio * rhs.numel())]
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
        ineq_state = None
        if self.has_ineq_constraint:
            ineq_state = self._compute_violations(
                x,
                self.A,
                self.b,
                self.A_sur,
                self.ineq_multiplier_type,
                self.ineq_observed_constraint_ratio,
            )
        eq_state = None
        if self.has_eq_constraint:
            eq_state = self._compute_violations(
                x, self.C, self.d, self.C_sur, self.eq_multiplier_type, self.eq_observed_constraint_ratio
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

    def compute_exact_solution(self):
        x = cp.Variable(self.num_variables)
        objective = cp.Minimize(cp.sum_squares(x))

        constraints = []
        if self.has_ineq_constraint:
            constraints.append(self.A.cpu().numpy() @ x <= self.b.cpu().numpy())
        if self.has_eq_constraint:
            constraints.append(self.C.cpu().numpy() @ x == self.d.cpu().numpy())

        prob = cp.Problem(objective, constraints)
        prob.solve()
        assert prob.status == cp.OPTIMAL

        x_star = torch.from_numpy(x.value).float().to(device=self.device)
        lambda_star = [torch.from_numpy(c.dual_value).float().to(device=self.device) for c in constraints]

        return x_star, lambda_star


def build_primal_optimizers(
    params: Iterable[torch.nn.Parameter],
    use_multiple_primal_optimizers=False,
    extrapolation=False,
    primal_optimizer_class=None,
    primal_optimizer_kwargs=None,
):
    if primal_optimizer_kwargs is None:
        primal_optimizer_kwargs = {"lr": 1e-2}

    if not use_multiple_primal_optimizers:
        if primal_optimizer_class is None:
            primal_optimizer_class = cooper.optim.ExtraSGD if extrapolation else torch.optim.SGD

        return primal_optimizer_class(params, **primal_optimizer_kwargs)

    if primal_optimizer_class is not None:
        primal_optimizer_class = [primal_optimizer_class] * 2
    else:
        if not extrapolation:
            primal_optimizer_class = [torch.optim.SGD, torch.optim.Adam]
        else:
            primal_optimizer_class = [cooper.optim.ExtraSGD, cooper.optim.ExtraAdam]

    primal_optimizers = []
    for i, param in enumerate(params):
        optimizer_class = primal_optimizer_class[i % 2]
        optimizer = optimizer_class([param], **primal_optimizer_kwargs)
        primal_optimizers.append(optimizer)

    return primal_optimizers


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


def build_cooper_optimizer(
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
