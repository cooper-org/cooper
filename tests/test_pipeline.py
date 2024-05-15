import pytest
import torch

import cooper
from cooper.penalty_coefficient_updaters import MultiplicativePenaltyCoefficientUpdater
from tests.helpers import cooper_test_utils

GROWTH_FACTOR = 1.002


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
        # pytest.skip("Indexed multipliers fail to converge in the Augmented Lagrangian formulation")
        # TODO(merajhashemi): Investigate why this is the case
        return cooper.multipliers.IndexedPenaltyCoefficient
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
        ineq_use_surrogate,
        ineq_multiplier_type,
        ineq_formulation_type,
        ineq_penalty_coefficient_type,
        device,
    ):
        self.cmp = cooper_test_utils.SquaredNormLinearCMP(
            has_ineq_constraint=True,
            has_eq_constraint=False,
            ineq_use_surrogate=ineq_use_surrogate,
            ineq_multiplier_type=ineq_multiplier_type,
            ineq_formulation_type=ineq_formulation_type,
            ineq_penalty_coefficient_type=ineq_penalty_coefficient_type,
            A=torch.eye(5, device=device),
            b=torch.zeros(5, device=device),
            device=device,
        )
        self.is_augmented_lagrangian = ineq_formulation_type == cooper.AugmentedLagrangianFormulation
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
            augmented_lagrangian=self.is_augmented_lagrangian,
            alternation_type=alternation_type,
        )

        penalty_updater = None
        if self.is_augmented_lagrangian:
            penalty_updater = MultiplicativePenaltyCoefficientUpdater(growth_factor=GROWTH_FACTOR)

        roll_kwargs = {"compute_cmp_state_kwargs": dict(x=x)}
        if not extrapolation and alternation_type == cooper_test_utils.AlternationType.PRIMAL_DUAL:
            roll_kwargs["compute_violations_kwargs"] = dict(x=x)

        for _ in range(2000):
            roll_out = cooper_optimizer.roll(**roll_kwargs)
            if self.is_augmented_lagrangian:
                penalty_updater.step(roll_out.cmp_state.observed_constraints)

        assert torch.allclose(x, torch.zeros_like(x), atol=8e-4)  # Tolerance is higher due to indexed multipliers
