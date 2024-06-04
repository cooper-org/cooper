import pytest
import torch

import cooper
from tests.helpers import cooper_test_utils


@pytest.fixture(scope="session", params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("Aim device 'cuda' is not available.")

    return torch.device(request.param)


@pytest.fixture(params=[cooper.ConstraintType.EQUALITY, cooper.ConstraintType.INEQUALITY])
def constraint_type(request):
    return request.param


@pytest.fixture(params=[True, False])
def use_multiple_primal_optimizers(request):
    return request.param


@pytest.fixture(params=[5])
def num_constraints(request):
    return request.param


@pytest.fixture(params=[5, 10])
def num_variables(request):
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
        pytest.skip("Extrapolation is not supported for Augmented Lagrangian formulation.")
    return request.param


@pytest.fixture(
    params=[
        cooper_test_utils.AlternationType.FALSE,
        cooper_test_utils.AlternationType.PRIMAL_DUAL,
        cooper_test_utils.AlternationType.DUAL_PRIMAL,
    ]
)
def alternation_type(request, extrapolation, formulation_type):
    if extrapolation and request.param != cooper_test_utils.AlternationType.FALSE:
        pytest.skip("Extrapolation is only supported for simultaneous updates.")
    if (
        formulation_type == cooper.AugmentedLagrangianFormulation
        and request.param == cooper_test_utils.AlternationType.FALSE
    ):
        pytest.skip("Augmented Lagrangian formulation requires alternation.")
    return request.param
