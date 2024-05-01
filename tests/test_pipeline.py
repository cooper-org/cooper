import pytest
import torch

import cooper
from tests.helpers import cooper_test_utils


@pytest.fixture(params=[True, False])
def has_eq_constraint(request):
    return request.param


@pytest.fixture(params=[True, False])
def ineq_use_surrogate(request):
    return request.param


@pytest.fixture(params=[cooper.multipliers.DenseMultiplier, cooper.multipliers.IndexedMultiplier])
def ineq_multiplier_type(request):
    return request.param


@pytest.fixture(params=[cooper.LagrangianFormulation, cooper.AugmentedLagrangianFormulation])
def ineq_formulation_type(request):
    return request.param


@pytest.fixture
def ineq_penalty_coefficient_type(ineq_formulation_type, ineq_multiplier_type):
    if ineq_formulation_type == cooper.LagrangianFormulation:
        return None
    if ineq_multiplier_type == cooper.multipliers.IndexedMultiplier:
        pytest.skip("Indexed multipliers fail to converge in the Augmented Lagrangian formulation")
        # TODO(merajhashemi): Investigate why this is the case
        # return cooper.multipliers.IndexedPenaltyCoefficient
    elif ineq_multiplier_type == cooper.multipliers.DenseMultiplier:
        return cooper.multipliers.DensePenaltyCoefficient


@pytest.fixture(params=[True, False])
def extrapolation(request, ineq_formulation_type):
    if ineq_formulation_type == cooper.AugmentedLagrangianFormulation:
        return False
    return request.param


@pytest.fixture(
    params=[
        cooper_test_utils.AlternationType.FALSE,
        cooper_test_utils.AlternationType.PRIMAL_DUAL,
        cooper_test_utils.AlternationType.DUAL_PRIMAL,
    ]
)
def alternation_type(request):
    return request.param


class TestConvergence:
    @pytest.fixture(autouse=True)
    def setup_cmp(
        self,
        has_eq_constraint,
        ineq_use_surrogate,
        ineq_multiplier_type,
        ineq_formulation_type,
        ineq_penalty_coefficient_type,
        device,
    ):
        self.cmp = cooper_test_utils.SquaredNormLinearCMP(
            has_ineq_constraint=True,
            has_eq_constraint=has_eq_constraint,
            ineq_use_surrogate=ineq_use_surrogate,
            ineq_multiplier_type=ineq_multiplier_type,
            ineq_formulation_type=ineq_formulation_type,
            ineq_penalty_coefficient_type=ineq_penalty_coefficient_type,
            A=torch.eye(5, device=device),
            b=torch.zeros(5, device=device),
            C=torch.eye(5, device=device),
            d=torch.zeros(5, device=device),
            device=device,
        )
        self.ineq_formulation_type = ineq_formulation_type
        self.device = device

    def test_convergence(self, extrapolation, alternation_type):
        x = torch.nn.Parameter(torch.ones(5, device=self.device))

        optimizer_class = cooper.optim.ExtraSGD if extrapolation else torch.optim.SGD
        primal_optimizers = optimizer_class([x], lr=1e-2)

        cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
            cmp=self.cmp,
            primal_optimizers=primal_optimizers,
            extrapolation=extrapolation,
            dual_optimizer_class=optimizer_class,
            augmented_lagrangian=self.ineq_formulation_type == cooper.AugmentedLagrangianFormulation,
            alternation_type=alternation_type,
            dual_optimizer_kwargs={"lr": 1e-2},
        )

        roll_kwargs = {"compute_cmp_state_kwargs": dict(x=x)}
        if not extrapolation and alternation_type == cooper_test_utils.AlternationType.PRIMAL_DUAL:
            roll_kwargs["compute_violations_kwargs"] = dict(x=x)

        for _ in range(2000):
            cooper_optimizer.roll(**roll_kwargs)

        assert torch.allclose(x, torch.zeros_like(x), atol=1e-3)  # Tolerance is higher due to indexed multipliers
