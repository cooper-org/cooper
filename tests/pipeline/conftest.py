import math

import pytest
import torch

import cooper
from cooper.penalty_coefficient_updaters import MultiplicativePenaltyCoefficientUpdater
from tests.helpers import cooper_test_utils

PRIMAL_LR = 3e-2
DUAL_LR = 2e-1
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
def use_surrogate(request):
    return request.param


@pytest.fixture(params=[cooper.multipliers.DenseMultiplier, cooper.multipliers.IndexedMultiplier])
def multiplier_type(request):
    return request.param


@pytest.fixture(params=[cooper.LagrangianFormulation, cooper.AugmentedLagrangianFormulation])
def formulation_type(request):
    return request.param


@pytest.fixture
def penalty_coefficient_type(formulation_type, multiplier_type):
    if formulation_type == cooper.LagrangianFormulation:
        return None
    if multiplier_type == cooper.multipliers.IndexedMultiplier:
        return cooper.multipliers.IndexedPenaltyCoefficient
    elif multiplier_type == cooper.multipliers.DenseMultiplier:
        return cooper.multipliers.DensePenaltyCoefficient


@pytest.fixture(params=[True, False])
def extrapolation(request, formulation_type):
    if request.param and formulation_type == cooper.AugmentedLagrangianFormulation:
        pytest.skip("Extrapolation is not supported for the Augmented Lagrangian formulation.")
    return request.param


@pytest.fixture(
    params=[
        cooper_test_utils.AlternationType.FALSE,
        cooper_test_utils.AlternationType.PRIMAL_DUAL,
        cooper_test_utils.AlternationType.DUAL_PRIMAL,
    ]
)
def alternation_type(request, extrapolation, formulation_type):
    is_alternation = request.param != cooper_test_utils.AlternationType.FALSE

    if extrapolation and is_alternation:
        pytest.skip("Extrapolation is only supported for simultaneous updates.")
    if formulation_type == cooper.AugmentedLagrangianFormulation and not is_alternation:
        pytest.skip("Augmented Lagrangian formulation requires alternation.")
    return request.param


@pytest.fixture
def unconstrained_cmp(device, num_variables):
    cmp = cooper_test_utils.SquaredNormLinearCMP(num_variables=num_variables, device=device)
    return cmp


@pytest.fixture
def params(device, num_variables, use_multiple_primal_optimizers):
    x_init = torch.ones(num_variables, device=device)
    x_init = x_init.tensor_split(2) if use_multiple_primal_optimizers else [x_init]
    params = list(map(lambda t: torch.nn.Parameter(t), x_init))
    return params


@pytest.fixture
def constraint_params(num_variables, num_constraints, seed, device):
    generator = torch.Generator(device).manual_seed(seed)

    # Uniform distribution between 1.5 and 2.5
    S = torch.diag(torch.rand(num_constraints, device=device, generator=generator) + 1.5)
    U, _ = torch.linalg.qr(torch.randn(num_constraints, num_constraints, device=device, generator=generator))
    V, _ = torch.linalg.qr(torch.randn(num_variables, num_variables, device=device, generator=generator))

    # Form the matrix U * S * V
    lhs = torch.mm(U, torch.mm(S, V[:num_constraints, :]))
    rhs = torch.randn(num_constraints, device=device, generator=generator)
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

    cmp_kwargs = dict(num_variables=num_variables, device=device)
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

    cmp = cooper_test_utils.SquaredNormLinearCMP(**cmp_kwargs)
    return cmp


@pytest.fixture
def cooper_optimizer_no_constraint(unconstrained_cmp, params):
    primal_optimizers = cooper_test_utils.build_primal_optimizers(
        params, primal_optimizer_kwargs=[{"lr": PRIMAL_LR} for _ in range(len(params))]
    )
    cooper_optimizer = cooper_test_utils.build_cooper_optimizer(
        cmp=unconstrained_cmp, primal_optimizers=primal_optimizers
    )
    return cooper_optimizer


@pytest.fixture
def cooper_optimizer(
    cmp, params, num_variables, use_multiple_primal_optimizers, extrapolation, alternation_type, formulation_type
):
    primal_optimizer_kwargs = [{"lr": PRIMAL_LR}]
    if use_multiple_primal_optimizers:
        primal_optimizer_kwargs.append({"lr": 10 * PRIMAL_LR, "betas": (0.0, 0.0), "eps": 10.0})
    primal_optimizers = cooper_test_utils.build_primal_optimizers(
        params, extrapolation, primal_optimizer_kwargs=primal_optimizer_kwargs
    )

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer(
        cmp=cmp,
        primal_optimizers=primal_optimizers,
        extrapolation=extrapolation,
        augmented_lagrangian=formulation_type == cooper.AugmentedLagrangianFormulation,
        alternation_type=alternation_type,
        dual_optimizer_class=cooper.optim.ExtraSGD if extrapolation else torch.optim.SGD,
        dual_optimizer_kwargs={"lr": DUAL_LR / math.sqrt(num_variables)},
    )
    return cooper_optimizer


@pytest.fixture
def penalty_updater(formulation_type):
    if formulation_type != cooper.AugmentedLagrangianFormulation:
        return None
    penalty_updater = MultiplicativePenaltyCoefficientUpdater(
        growth_factor=PENALTY_GROWTH_FACTOR, violation_tolerance=PENALTY_VIOLATION_TOLERANCE
    )
    return penalty_updater
