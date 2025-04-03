import math

import pytest
import torch

import cooper
import testing

PRIMAL_LR = 3e-2
PRIMAL_LR_QUADRATIC_PENALTY = 1e-4
DUAL_LR = 2e-1
PENALTY_INCREMENT = 1.75
PENALTY_GROWTH_FACTOR = 1.0 + 2.5e-4
PENALTY_VIOLATION_TOLERANCE = 1e-4


@pytest.fixture(params=[0, 1])
def seed(request):
    return request.param


@pytest.fixture(params=[cooper.ConstraintType.EQUALITY, cooper.ConstraintType.INEQUALITY])
def constraint_type(request):
    return request.param


@pytest.fixture(params=[True, False])
def use_multiple_primal_optimizers(request):
    return request.param


@pytest.fixture(params=[5, 10])
def num_variables(request):
    return request.param


@pytest.fixture(params=[5])
def num_constraints(request, num_variables):
    if request.param > num_variables:
        pytest.skip("Overconstrained problem. Skipping test.")
    return request.param


@pytest.fixture(params=[True, False])
def use_surrogate(request, formulation_type):
    if request.param and formulation_type == cooper.formulations.QuadraticPenalty:
        pytest.skip("Proxy constraints are not applicable for the Quadratic Penalty formulation.")
    return request.param


@pytest.fixture(
    params=[
        cooper.formulations.Lagrangian,
        cooper.formulations.QuadraticPenalty,
        cooper.formulations.AugmentedLagrangian,
    ]
)
def formulation_type(request):
    return request.param


@pytest.fixture(
    params=[
        (cooper.multipliers.DenseMultiplier, cooper.penalty_coefficients.DensePenaltyCoefficient),
        (cooper.multipliers.IndexedMultiplier, cooper.penalty_coefficients.IndexedPenaltyCoefficient),
    ]
)
def multiplier_penalty_coefficient_types(request, formulation_type):
    multiplier, penalty_coefficient = request.param
    if formulation_type == cooper.formulations.Lagrangian:
        return multiplier, None
    if formulation_type == cooper.formulations.QuadraticPenalty:
        return None, penalty_coefficient
    return multiplier, penalty_coefficient


@pytest.fixture
def multiplier_type(multiplier_penalty_coefficient_types):
    multiplier, _ = multiplier_penalty_coefficient_types
    return multiplier


@pytest.fixture
def penalty_coefficient_type(multiplier_penalty_coefficient_types):
    _, penalty_coefficient = multiplier_penalty_coefficient_types
    return penalty_coefficient


@pytest.fixture(params=[True, False])
def extrapolation(request, formulation_type):
    if request.param and formulation_type == cooper.formulations.QuadraticPenalty:
        pytest.skip("Extrapolation is not supported for the Quadratic Penalty formulation.")
    return request.param


@pytest.fixture(
    params=[testing.AlternationType.FALSE, testing.AlternationType.PRIMAL_DUAL, testing.AlternationType.DUAL_PRIMAL]
)
def alternation_type(request, extrapolation, formulation_type):
    is_alternation = request.param != testing.AlternationType.FALSE

    if extrapolation and is_alternation:
        pytest.skip("Extrapolation is only supported for simultaneous updates.")
    if formulation_type == cooper.formulations.QuadraticPenalty and is_alternation:
        pytest.skip("Quadratic Penalty formulation does not support alternation.")
    return request.param


@pytest.fixture
def unconstrained_cmp(device, num_variables):
    cmp = testing.SquaredNormLinearCMP(num_variables=num_variables, device=device)
    return cmp


@pytest.fixture
def params(device, num_variables, use_multiple_primal_optimizers):
    x_init = torch.ones(num_variables, device=device)
    x_init = x_init.tensor_split(2) if use_multiple_primal_optimizers else [x_init]
    params = [torch.nn.Parameter(t) for t in x_init]
    return params


@pytest.fixture
def constraint_params(num_variables, num_constraints, seed, device):
    generator = torch.Generator().manual_seed(seed)

    # Uniform distribution between 1.5 and 2.5
    S = torch.diag(torch.rand(num_constraints, generator=generator).to(device) + 1.5)
    U, _ = torch.linalg.qr(torch.randn(num_constraints, num_constraints, generator=generator).to(device))
    V, _ = torch.linalg.qr(torch.randn(num_variables, num_variables, generator=generator).to(device))

    # Form the matrix U * S * V
    lhs = torch.mm(U, torch.mm(S, V[:num_constraints, :]))
    rhs = torch.randn(num_constraints, generator=generator).to(device)
    rhs = rhs / rhs.norm()

    return lhs, rhs


@pytest.fixture
def cmp(
    constraint_params,
    constraint_type,
    num_variables,
    use_surrogate,
    multiplier_type,
    formulation_type,
    penalty_coefficient_type,
    device,
):
    lhs, rhs = constraint_params

    cmp_kwargs = {"num_variables": num_variables, "device": device}
    is_inequality = constraint_type == cooper.ConstraintType.INEQUALITY
    if is_inequality:
        cmp_kwargs["A"] = lhs
        cmp_kwargs["b"] = rhs
        prefix = "ineq"
    else:
        cmp_kwargs["C"] = lhs
        cmp_kwargs["d"] = rhs
        prefix = "eq"

    cmp_kwargs[f"has_{prefix}_constraint"] = True
    cmp_kwargs[f"{prefix}_use_surrogate"] = use_surrogate
    cmp_kwargs[f"{prefix}_multiplier_type"] = multiplier_type
    cmp_kwargs[f"{prefix}_formulation_type"] = formulation_type
    cmp_kwargs[f"{prefix}_penalty_coefficient_type"] = penalty_coefficient_type

    cmp = testing.SquaredNormLinearCMP(**cmp_kwargs)
    return cmp


@pytest.fixture
def cooper_optimizer_no_constraint(unconstrained_cmp, params):
    primal_optimizers = testing.build_primal_optimizers(
        params, primal_optimizer_kwargs=[{"lr": PRIMAL_LR} for _ in range(len(params))]
    )
    cooper_optimizer = testing.build_cooper_optimizer(cmp=unconstrained_cmp, primal_optimizers=primal_optimizers)
    return cooper_optimizer


@pytest.fixture
def primal_lr(formulation_type):
    if formulation_type == cooper.formulations.QuadraticPenalty:
        return PRIMAL_LR_QUADRATIC_PENALTY
    return PRIMAL_LR


@pytest.fixture
def dual_lr(num_variables):
    return DUAL_LR / math.sqrt(num_variables)


@pytest.fixture
def cooper_optimizer(cmp, params, primal_lr, dual_lr, use_multiple_primal_optimizers, extrapolation, alternation_type):
    primal_optimizer_kwargs = [{"lr": primal_lr}]
    if use_multiple_primal_optimizers:
        primal_optimizer_kwargs.append({"lr": 10 * primal_lr, "betas": (0.0, 0.0), "eps": 10.0})
    primal_optimizers = testing.build_primal_optimizers(
        params, extrapolation, primal_optimizer_kwargs=primal_optimizer_kwargs
    )

    cooper_optimizer = testing.build_cooper_optimizer(
        cmp=cmp,
        primal_optimizers=primal_optimizers,
        extrapolation=extrapolation,
        alternation_type=alternation_type,
        dual_optimizer_class=cooper.optim.ExtraSGD if extrapolation else torch.optim.SGD,
        dual_optimizer_kwargs={"lr": dual_lr},
    )
    return cooper_optimizer


@pytest.fixture
def penalty_updater(formulation_type):
    if formulation_type == cooper.formulations.QuadraticPenalty:
        return cooper.penalty_coefficients.AdditivePenaltyCoefficientUpdater(
            increment=PENALTY_INCREMENT, violation_tolerance=PENALTY_VIOLATION_TOLERANCE
        )
    if formulation_type == cooper.formulations.AugmentedLagrangian:
        return cooper.penalty_coefficients.MultiplicativePenaltyCoefficientUpdater(
            growth_factor=PENALTY_GROWTH_FACTOR, violation_tolerance=PENALTY_VIOLATION_TOLERANCE
        )
    return None
