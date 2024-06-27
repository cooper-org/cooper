import pytest
import torch

import cooper
from tests.helpers import cooper_test_utils


@pytest.fixture(scope="session", params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("Aim device 'cuda' is not available.")
    return torch.device(request.param)


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
def cmp_no_constraint(device, num_variables):
    cmp = cooper_test_utils.SquaredNormLinearCMP(num_variables=num_variables, device=device)
    return cmp


@pytest.fixture
def params(device, num_variables, use_multiple_primal_optimizers):
    x_init = torch.ones(num_variables, device=device)
    x_init = x_init.tensor_split(2) if use_multiple_primal_optimizers else [x_init]
    params = list(map(lambda t: torch.nn.Parameter(t), x_init))
    return params
