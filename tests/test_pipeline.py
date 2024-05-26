from typing import Optional

import pytest
import torch

import cooper
from cooper.penalty_coefficient_updaters import MultiplicativePenaltyCoefficientUpdater
from tests.helpers import cooper_test_utils

PRIMAL_LR = 1e-2
DUAL_LR = 1e-2
PENALTY_GROWTH_FACTOR = 1.002
PENALTY_VIOLATION_TOLERANCE = 1e-4


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
    if request.param and ineq_formulation_type == cooper.AugmentedLagrangianFormulation:
        pytest.skip("Extrapolation is not supported for Augmented Lagrangian formulation.")
    return request.param


@pytest.fixture(
    params=[
        cooper_test_utils.AlternationType.FALSE,
        cooper_test_utils.AlternationType.PRIMAL_DUAL,
        cooper_test_utils.AlternationType.DUAL_PRIMAL,
    ]
)
def alternation_type(request, extrapolation, ineq_formulation_type):
    if extrapolation and request.param != cooper_test_utils.AlternationType.FALSE:
        pytest.skip("Extrapolation is only supported for simultaneous updates.")
    if (
        ineq_formulation_type == cooper.AugmentedLagrangianFormulation
        and request.param == cooper_test_utils.AlternationType.FALSE
    ):
        pytest.skip("Augmented Lagrangian formulation requires alternation.")
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
        self.is_indexed_multiplier = ineq_multiplier_type == cooper.multipliers.IndexedMultiplier
        self.num_variables = num_variables
        self.num_constraints = num_constraints
        self.device = device

    def test_convergence(self, extrapolation, alternation_type):
        if self.num_constraints > self.num_variables:
            pytest.skip("Overconstrained problem. Skipping test.")

        x = torch.nn.Parameter(torch.ones(self.num_variables, device=self.device))

        optimizer_class = cooper.optim.ExtraSGD if extrapolation else torch.optim.SGD
        primal_optimizers = optimizer_class([x], lr=PRIMAL_LR)

        cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
            cmp=self.cmp,
            primal_optimizers=primal_optimizers,
            extrapolation=extrapolation,
            dual_optimizer_class=optimizer_class,
            augmented_lagrangian=self.is_augmented_lagrangian,
            alternation_type=alternation_type,
            dual_optimizer_kwargs={"lr": DUAL_LR},
        )

        penalty_updater = None
        if self.is_augmented_lagrangian:
            penalty_updater = MultiplicativePenaltyCoefficientUpdater(
                growth_factor=PENALTY_GROWTH_FACTOR, violation_tolerance=PENALTY_VIOLATION_TOLERANCE
            )

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
        atol = 1e-5 if not self.ineq_use_surrogate else 1e-3
        assert torch.allclose(x, x_star, atol=atol)

        # Check if the dual variable is close to the exact solution
        assert torch.allclose(list(self.cmp.dual_parameters())[0].view(-1), lambda_star[0], atol=atol)

    def test_manual_step(self, extrapolation, alternation_type):

        if alternation_type == cooper_test_utils.AlternationType.PRIMAL_DUAL and self.is_indexed_multiplier:
            pytest.skip("Cannot test IndexedMultiplier with PRIMAL_DUAL alternation.")

        x = torch.nn.Parameter(torch.ones(self.num_variables, device=self.device))
        optimizer_class = cooper.optim.ExtraSGD if extrapolation else torch.optim.SGD
        primal_optimizers = optimizer_class([x], lr=PRIMAL_LR)

        cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
            cmp=self.cmp,
            primal_optimizers=primal_optimizers,
            extrapolation=extrapolation,
            dual_optimizer_class=optimizer_class,
            augmented_lagrangian=self.is_augmented_lagrangian,
            alternation_type=alternation_type,
            dual_optimizer_kwargs={"lr": DUAL_LR},
        )

        penalty_updater = None
        if self.is_augmented_lagrangian:
            penalty_updater = MultiplicativePenaltyCoefficientUpdater(
                growth_factor=PENALTY_GROWTH_FACTOR, violation_tolerance=PENALTY_VIOLATION_TOLERANCE
            )

        roll_kwargs = {"compute_cmp_state_kwargs": dict(x=x)}
        if not extrapolation and alternation_type == cooper_test_utils.AlternationType.PRIMAL_DUAL:
            roll_kwargs["compute_violations_kwargs"] = dict(x=x)

        manual_x = torch.ones(self.num_variables, device=self.device)
        manual_multiplier = torch.zeros(self.num_constraints, device=self.device)
        manual_penalty_coeff = None
        if self.is_augmented_lagrangian:
            manual_penalty_coeff = torch.ones(self.num_constraints, device=self.device)

        # ----------------------- First iteration -----------------------
        roll_out = cooper_optimizer.roll(**roll_kwargs)
        if self.is_augmented_lagrangian:
            penalty_updater.step(roll_out.cmp_state.observed_constraints)

        if alternation_type == cooper_test_utils.AlternationType.PRIMAL_DUAL:
            observed_multipliers = torch.cat(list(roll_out.dual_lagrangian_store.observed_multiplier_values()))
        else:
            observed_multipliers = torch.cat(list(roll_out.primal_lagrangian_store.observed_multiplier_values()))

        features = list(roll_out.cmp_state.observed_constraints.values())[0].constraint_features
        if features is None:
            features = torch.arange(self.num_constraints, device=self.device, dtype=torch.long)

        strict_features = list(roll_out.cmp_state.observed_constraints.values())[0].strict_constraint_features
        if strict_features is None:
            strict_features = torch.arange(self.num_constraints, device=self.device, dtype=torch.long)

        # Manual step
        (
            manual_x,
            manual_multiplier,
            manual_primal_lagrangian,
            manual_dual_lagrangian,
            manual_observed_multipliers,
        ) = self.manual_roll(
            manual_x,
            manual_multiplier,
            features,
            strict_features,
            alternation_type,
            manual_penalty_coeff,
            extrapolation,
        )

        # Check manual and cooper outputs are close
        assert torch.allclose(observed_multipliers, manual_observed_multipliers[features], atol=1e-4)
        assert torch.allclose(x, manual_x, atol=1e-4)
        assert torch.allclose(roll_out.primal_lagrangian_store.lagrangian, manual_primal_lagrangian, atol=1e-4)
        assert torch.allclose(roll_out.dual_lagrangian_store.lagrangian, manual_dual_lagrangian, atol=1e-4)

        if self.is_augmented_lagrangian:
            positive_violations = (self.cmp.A @ manual_x - self.cmp.b)[strict_features].relu()
            violated_indices = positive_violations > PENALTY_VIOLATION_TOLERANCE
            # Update the penalty coefficients for the violated constraints
            manual_penalty_coeff[strict_features[violated_indices]] *= PENALTY_GROWTH_FACTOR

        # ----------------------- Second iteration -----------------------
        roll_out = cooper_optimizer.roll(**roll_kwargs)
        if self.is_augmented_lagrangian:
            penalty_updater.step(roll_out.cmp_state.observed_constraints)

        if alternation_type == cooper_test_utils.AlternationType.PRIMAL_DUAL:
            observed_multipliers = torch.cat(list(roll_out.dual_lagrangian_store.observed_multiplier_values()))
        else:
            observed_multipliers = torch.cat(list(roll_out.primal_lagrangian_store.observed_multiplier_values()))

        features = list(roll_out.cmp_state.observed_constraints.values())[0].constraint_features
        if features is None:
            features = torch.arange(self.num_constraints, device=self.device, dtype=torch.long)

        strict_features = list(roll_out.cmp_state.observed_constraints.values())[0].strict_constraint_features
        if strict_features is None:
            strict_features = torch.arange(self.num_constraints, device=self.device, dtype=torch.long)

        # Manual step
        (
            manual_x,
            manual_multiplier,
            manual_primal_lagrangian,
            manual_dual_lagrangian,
            manual_observed_multipliers,
        ) = self.manual_roll(
            manual_x,
            manual_multiplier,
            features,
            strict_features,
            alternation_type,
            manual_penalty_coeff,
            extrapolation,
        )

        # Check manual and cooper outputs are close
        assert torch.allclose(observed_multipliers, manual_observed_multipliers[features], atol=1e-4)
        assert torch.allclose(x, manual_x, atol=1e-4)
        assert torch.allclose(roll_out.primal_lagrangian_store.lagrangian, manual_primal_lagrangian, atol=1e-4)
        assert torch.allclose(roll_out.dual_lagrangian_store.lagrangian, manual_dual_lagrangian, atol=1e-4)

        if self.is_augmented_lagrangian:
            positive_violations = (self.cmp.A @ manual_x - self.cmp.b)[strict_features].relu()
            violated_indices = positive_violations > PENALTY_VIOLATION_TOLERANCE
            # Update the penalty coefficients for the violated constraints
            manual_penalty_coeff[strict_features[violated_indices]] *= PENALTY_GROWTH_FACTOR

    def _manual_violation(self, manual_x, strict=False):
        if strict and self.ineq_use_surrogate:
            return self.cmp.A_sur @ manual_x - self.cmp.b
        return self.cmp.A @ manual_x - self.cmp.b

    def _manual_primal_step(self, manual_x, manual_multiplier, features, manual_penalty_coeff=None):
        if manual_penalty_coeff is not None:
            # The gradient of the Augmented Lagrangian wrt the primal variables for inequality
            # constraints is:
            #  grad_objective + grad_constraint * relu(multiplier + penalty_coeff * violation)
            violation = self._manual_violation(manual_x)[features]
            manual_x_grad = 2 * manual_x + self.cmp.A[features].t() @ torch.relu(
                manual_multiplier[features] + manual_penalty_coeff[features] * violation
            )
        else:
            manual_x_grad = 2 * manual_x + self.cmp.A[features].t() @ manual_multiplier[features]
        manual_x = manual_x - PRIMAL_LR * manual_x_grad
        return manual_x

    def _manual_dual_step(self, manual_x, manual_multiplier, strict_features, manual_penalty_coeff=None):
        violation = self._manual_violation(manual_x, strict=True)[strict_features]
        if manual_penalty_coeff is None:
            manual_multiplier[strict_features] = torch.relu(manual_multiplier[strict_features] + DUAL_LR * violation)
        else:
            manual_multiplier[strict_features] = torch.relu(
                manual_multiplier[strict_features] + manual_penalty_coeff[strict_features] * violation
            )
        return manual_multiplier

    def _manual_primal_lagrangian(
        self, manual_x, manual_multiplier, features, manual_penalty_coeff: Optional[torch.Tensor] = None
    ):
        violation = self._manual_violation(manual_x)[features]
        loss = torch.sum(manual_x**2)

        if manual_penalty_coeff is None:
            return loss + torch.dot(manual_multiplier[features], violation)

        aug_lag = loss + 0.5 * torch.dot(
            1 / manual_penalty_coeff[features],
            torch.relu(manual_multiplier[features] + manual_penalty_coeff[features] * violation) ** 2
            - manual_multiplier[features] ** 2,
        )
        return aug_lag

    def _manual_dual_lagrangian(self, manual_x, manual_multiplier, strict_features, manual_penalty_coeff=None):
        violation = self._manual_violation(manual_x, strict=True)[strict_features]
        if manual_penalty_coeff is None:
            return torch.sum(manual_multiplier[strict_features] * violation)
        return torch.sum(manual_penalty_coeff[strict_features] * manual_multiplier[strict_features] * violation)

    def _manual_simultaneous_roll(self, manual_x, manual_multiplier, features, strict_features):
        manual_primal_lagrangian = self._manual_primal_lagrangian(manual_x, manual_multiplier, features)
        manual_observed_multipliers = manual_multiplier.clone()
        manual_dual_lagrangian = self._manual_dual_lagrangian(manual_x, manual_multiplier, strict_features)
        manual_x, manual_multiplier = self._manual_primal_step(
            manual_x, manual_multiplier, features
        ), self._manual_dual_step(manual_x, manual_multiplier, strict_features)
        return (
            manual_x,
            manual_multiplier,
            manual_primal_lagrangian,
            manual_dual_lagrangian,
            manual_observed_multipliers,
        )

    def _manual_dual_primal_roll(self, manual_x, manual_multiplier, features, strict_features, manual_penalty_coeff):
        manual_dual_lagrangian = self._manual_dual_lagrangian(
            manual_x, manual_multiplier, strict_features, manual_penalty_coeff
        )
        manual_multiplier = self._manual_dual_step(manual_x, manual_multiplier, strict_features, manual_penalty_coeff)
        manual_primal_lagrangian = self._manual_primal_lagrangian(
            manual_x, manual_multiplier, features, manual_penalty_coeff
        )
        manual_observed_multipliers = manual_multiplier.clone()
        manual_x = self._manual_primal_step(manual_x, manual_multiplier, features, manual_penalty_coeff)
        return (
            manual_x,
            manual_multiplier,
            manual_primal_lagrangian,
            manual_dual_lagrangian,
            manual_observed_multipliers,
        )

    def _manual_primal_dual_roll(self, manual_x, manual_multiplier, features, strict_features, manual_penalty_coeff):
        manual_primal_lagrangian = self._manual_primal_lagrangian(
            manual_x, manual_multiplier, features, manual_penalty_coeff
        )
        manual_observed_multipliers = manual_multiplier.clone()
        manual_x = self._manual_primal_step(manual_x, manual_multiplier, features, manual_penalty_coeff)
        manual_dual_lagrangian = self._manual_dual_lagrangian(
            manual_x, manual_multiplier, strict_features, manual_penalty_coeff
        )
        manual_multiplier = self._manual_dual_step(manual_x, manual_multiplier, strict_features, manual_penalty_coeff)
        return (
            manual_x,
            manual_multiplier,
            manual_primal_lagrangian,
            manual_dual_lagrangian,
            manual_observed_multipliers,
        )

    def _manual_extragradient_roll(self, manual_x, manual_multiplier, features, strict_features):
        manual_x_copy = manual_x.clone()
        manual_multiplier_copy = manual_multiplier.clone()
        manual_x = self._manual_primal_step(manual_x_copy, manual_multiplier_copy, features)
        manual_multiplier = self._manual_dual_step(manual_x_copy, manual_multiplier_copy.clone(), strict_features)

        manual_primal_lagrangian = self._manual_primal_lagrangian(manual_x, manual_multiplier, features)
        manual_observed_multipliers = manual_multiplier.clone()
        manual_dual_lagrangian = self._manual_dual_lagrangian(manual_x, manual_multiplier, strict_features)

        manual_x_grad = 2 * manual_x + self.cmp.A[features].t() @ manual_multiplier[features]
        manual_x, manual_multiplier = manual_x_copy - PRIMAL_LR * manual_x_grad, self._manual_dual_step(
            manual_x, manual_multiplier_copy, strict_features
        )

        return (
            manual_x,
            manual_multiplier,
            manual_primal_lagrangian,
            manual_dual_lagrangian,
            manual_observed_multipliers,
        )

    @torch.inference_mode()
    def manual_roll(
        self,
        manual_x,
        manual_multiplier,
        features,
        strict_features,
        alternation_type,
        manual_penalty_coeff,
        extrapolation,
    ):
        if extrapolation:
            return self._manual_extragradient_roll(manual_x, manual_multiplier, features, strict_features)

        if alternation_type == cooper_test_utils.AlternationType.FALSE:
            return self._manual_simultaneous_roll(manual_x, manual_multiplier, features, strict_features)
        elif alternation_type == cooper_test_utils.AlternationType.DUAL_PRIMAL:
            return self._manual_dual_primal_roll(
                manual_x, manual_multiplier, features, strict_features, manual_penalty_coeff
            )
        elif alternation_type == cooper_test_utils.AlternationType.PRIMAL_DUAL:
            return self._manual_primal_dual_roll(
                manual_x, manual_multiplier, features, strict_features, manual_penalty_coeff
            )
        else:
            raise ValueError(f"Unknown alternation type: {alternation_type}")
