import pytest
import torch

import cooper
from cooper.penalty_coefficient_updaters import MultiplicativePenaltyCoefficientUpdater
from tests.helpers import cooper_test_utils

GROWTH_FACTOR = 1.002


@pytest.fixture(params=[5])
def num_constraints(request):
    return request.param


@pytest.fixture(params=[5, 10])
def num_variables(request):
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
        num_constraints,
        num_variables,
        device,
    ):

        A = torch.randn(
            num_constraints, num_variables, device=device, generator=torch.Generator(device=device).manual_seed(0)
        )
        b = torch.randn(num_constraints, device=device, generator=torch.Generator(device=device).manual_seed(0))

        self.cmp = cooper_test_utils.SquaredNormLinearCMP(
            has_ineq_constraint=True,
            has_eq_constraint=False,
            ineq_use_surrogate=ineq_use_surrogate,
            ineq_multiplier_type=ineq_multiplier_type,
            ineq_formulation_type=ineq_formulation_type,
            ineq_penalty_coefficient_type=ineq_penalty_coefficient_type,
            A=A,
            b=b,
            device=device,
        )

        self.ineq_use_surrogate = ineq_use_surrogate
        self.is_augmented_lagrangian = ineq_formulation_type == cooper.AugmentedLagrangianFormulation
        self.num_variables = num_variables
        self.device = device

    def test_convergence(self, extrapolation, alternation_type):
        x = torch.nn.Parameter(torch.ones(self.num_variables, device=self.device))

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

        # Compute the exact solution
        x_star, lambda_star = self.cmp.compute_exact_solution()

        # Check if the primal variable is close to the exact solution
        # The tolerance is higher for the surrogate case
        atol = 1e-5 if not self.ineq_use_surrogate else 1e-2
        assert torch.allclose(x, x_star, atol=atol)

        # Check if the dual variable is close to the exact solution
        assert torch.allclose(list(self.cmp.dual_parameters())[0].view(-1), lambda_star[0], atol=atol)
