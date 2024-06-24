import math
from typing import Optional

import pytest
import torch

import cooper
from cooper.penalty_coefficient_updaters import MultiplicativePenaltyCoefficientUpdater
from tests.helpers import cooper_test_utils

PRIMAL_LR = 3e-2
DUAL_LR = 2e-1
PENALTY_GROWTH_FACTOR = 1.0 + 2.5e-4
PENALTY_VIOLATION_TOLERANCE = 1e-4


class TestConvergenceNoConstraint:
    @pytest.fixture(autouse=True)
    def setup_cmp(self, num_variables, device):
        self.cmp = cooper_test_utils.SquaredNormLinearCMP(num_variables=num_variables, device=device)
        self.num_variables = num_variables
        self.device = device

    def test_convergence_no_constraint(self, use_multiple_primal_optimizers):
        x_init = torch.ones(self.num_variables, device=self.device)
        x_init = x_init.tensor_split(2) if use_multiple_primal_optimizers else [x_init]
        params = list(map(lambda t: torch.nn.Parameter(t), x_init))
        primal_optimizers = cooper_test_utils.build_primal_optimizers(
            params, primal_optimizer_kwargs=[{"lr": PRIMAL_LR} for _ in range(len(params))]
        )

        cooper_optimizer = cooper_test_utils.build_cooper_optimizer(cmp=self.cmp, primal_optimizers=primal_optimizers)

        for _ in range(2000):
            cooper_optimizer.roll(compute_cmp_state_kwargs=dict(x=torch.cat(params)))

        # Compute the exact solution
        x_star, lambda_star = self.cmp.compute_exact_solution()

        # Check if the primal variable is close to the exact solution
        assert torch.allclose(torch.cat(params), x_star, atol=1e-5)


class TestConvergence:
    @pytest.fixture(autouse=True)
    def setup_cmp(
        self,
        constraint_type,
        use_surrogate,
        multiplier_type,
        formulation_type,
        penalty_coefficient_type,
        num_constraints,
        num_variables,
        seed,
        device,
    ):

        generator = torch.Generator(device).manual_seed(seed)

        # Uniform distribution between 1.5 and 2.5
        S = torch.diag(torch.rand(num_constraints, device=device, generator=generator) + 1.5)
        U, _ = torch.linalg.qr(torch.randn(num_constraints, num_constraints, device=device, generator=generator))
        V, _ = torch.linalg.qr(torch.randn(num_variables, num_variables, device=device, generator=generator))
        # Form the matrix U * S * V
        self.lhs = torch.mm(U, torch.mm(S, V[:num_constraints, :]))
        self.rhs = torch.randn(num_constraints, device=device, generator=generator)
        self.rhs = self.rhs / self.rhs.norm()

        self.is_inequality = constraint_type == cooper.ConstraintType.INEQUALITY

        cmp_kwargs = dict(num_variables=num_variables, device=device)
        if self.is_inequality:
            cmp_kwargs["A"] = self.lhs
            cmp_kwargs["b"] = self.rhs
            prefix = "ineq"

        else:
            cmp_kwargs["C"] = self.lhs
            cmp_kwargs["d"] = self.rhs
            prefix = "eq"

        cmp_kwargs[f"has_{prefix}_constraint"] = True
        cmp_kwargs[f"{prefix}_use_surrogate"] = use_surrogate
        cmp_kwargs[f"{prefix}_multiplier_type"] = multiplier_type
        cmp_kwargs[f"{prefix}_formulation_type"] = formulation_type
        cmp_kwargs[f"{prefix}_penalty_coefficient_type"] = penalty_coefficient_type

        self.cmp = cooper_test_utils.SquaredNormLinearCMP(**cmp_kwargs)
        self.lhs_sur = self.cmp.A_sur if self.is_inequality else self.cmp.C_sur

        self.constraint_type = constraint_type
        self.use_surrogate = use_surrogate
        self.is_augmented_lagrangian = formulation_type == cooper.AugmentedLagrangianFormulation
        self.is_indexed_multiplier = multiplier_type == cooper.multipliers.IndexedMultiplier
        self.num_variables = num_variables
        self.num_constraints = num_constraints
        self.device = device
        self.primal_lr = PRIMAL_LR
        self.dual_lr = DUAL_LR / math.sqrt(num_variables)

    def test_convergence(self, extrapolation, alternation_type, use_multiple_primal_optimizers):

        x_init = torch.ones(self.num_variables, device=self.device)
        x_init = x_init.tensor_split(2) if use_multiple_primal_optimizers else [x_init]
        params = list(map(lambda t: torch.nn.Parameter(t), x_init))

        primal_optimizer_kwargs = [{"lr": self.primal_lr}]
        if use_multiple_primal_optimizers:
            primal_optimizer_kwargs.append({"lr": 10 * self.primal_lr, "betas": (0.0, 0.0), "eps": 10.0})
        primal_optimizers = cooper_test_utils.build_primal_optimizers(
            params, extrapolation, primal_optimizer_kwargs=primal_optimizer_kwargs
        )

        cooper_optimizer = cooper_test_utils.build_cooper_optimizer(
            cmp=self.cmp,
            primal_optimizers=primal_optimizers,
            extrapolation=extrapolation,
            augmented_lagrangian=self.is_augmented_lagrangian,
            alternation_type=alternation_type,
            dual_optimizer_class=cooper.optim.ExtraSGD if extrapolation else torch.optim.SGD,
            dual_optimizer_kwargs={"lr": self.dual_lr},
        )

        penalty_updater = None
        if self.is_augmented_lagrangian:
            penalty_updater = MultiplicativePenaltyCoefficientUpdater(
                growth_factor=PENALTY_GROWTH_FACTOR, violation_tolerance=PENALTY_VIOLATION_TOLERANCE
            )

        for _ in range(2000):
            roll_kwargs = {"compute_cmp_state_kwargs": dict(x=torch.cat(params))}
            if alternation_type == cooper_test_utils.AlternationType.PRIMAL_DUAL:
                roll_kwargs["compute_violations_kwargs"] = dict(x=torch.cat(params))

            roll_out = cooper_optimizer.roll(**roll_kwargs)
            if self.is_augmented_lagrangian:
                penalty_updater.step(roll_out.cmp_state.observed_constraints)

        # Compute the exact solution
        x_star, lambda_star = self.cmp.compute_exact_solution()

        if not self.use_surrogate:
            # Check if the primal variable is close to the exact solution
            atol = 1e-4
            assert torch.allclose(torch.cat(params), x_star, atol=atol)

            # Check if the dual variable is close to the exact solution
            assert torch.allclose(list(self.cmp.dual_parameters())[0].view(-1), lambda_star[0], atol=atol)
        else:
            # The surrogate formulation is not guaranteed to converge to the exact solution,
            # but it should be feasible
            atol = 5e-4
            assert torch.le(self.lhs @ torch.cat(params) - self.rhs, atol).all()

    def test_manual_step(self, extrapolation, alternation_type):
        """
        Test the cooper optimizer roll methods implementation.

        This method tests the cooper optimizers by comparing the results with the manual implementation.
        The manual implementation assumes Stochastic Gradient Descent (SGD) is used for both the primal
        and dual optimizers.
        """

        if alternation_type == cooper_test_utils.AlternationType.PRIMAL_DUAL and self.is_indexed_multiplier:
            pytest.skip("Cannot test IndexedMultiplier with PRIMAL_DUAL alternation.")

        x = torch.nn.Parameter(torch.ones(self.num_variables, device=self.device))
        optimizer_class = cooper.optim.ExtraSGD if extrapolation else torch.optim.SGD
        primal_optimizers = optimizer_class([x], lr=PRIMAL_LR)

        cooper_optimizer = cooper_test_utils.build_cooper_optimizer(
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
        if alternation_type == cooper_test_utils.AlternationType.PRIMAL_DUAL:
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

        manual_x_prev = manual_x.clone()
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
        assert torch.allclose(observed_multipliers, manual_observed_multipliers[features])
        assert torch.allclose(x, manual_x)
        assert torch.allclose(roll_out.primal_lagrangian_store.lagrangian, manual_primal_lagrangian)
        assert torch.allclose(roll_out.dual_lagrangian_store.lagrangian, manual_dual_lagrangian)

        if self.is_augmented_lagrangian:
            # Update penalty coefficients for Augmented Lagrangian Method
            self._manual_update_penalty_coefficients(
                manual_x, manual_x_prev, strict_features, alternation_type, manual_penalty_coeff
            )

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

        manual_x_prev = manual_x.clone()
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
        assert torch.allclose(observed_multipliers, manual_observed_multipliers[features])
        assert torch.allclose(x, manual_x)
        assert torch.allclose(roll_out.primal_lagrangian_store.lagrangian, manual_primal_lagrangian)
        assert torch.allclose(roll_out.dual_lagrangian_store.lagrangian, manual_dual_lagrangian)

        if self.is_augmented_lagrangian:
            # Update penalty coefficients for Augmented Lagrangian Method
            self._manual_update_penalty_coefficients(
                manual_x, manual_x_prev, strict_features, alternation_type, manual_penalty_coeff
            )

    def _manual_violation(self, manual_x, strict=False):
        """
        Compute the constraint violations given the primal variables.
        If strict is True, the strict violations are computed.
        Otherwise, the surrogate violations are computed if the surrogates are provided.
        """
        if not strict and self.use_surrogate:
            return self.lhs_sur @ manual_x - self.rhs
        return self.lhs @ manual_x - self.rhs

    def _manual_primal_step(self, manual_x, manual_multiplier, features, manual_penalty_coeff=None):
        lhs = self.lhs_sur if self.use_surrogate else self.lhs
        obj_grad = 2 * manual_x
        if manual_penalty_coeff is not None:
            # The gradient of the Augmented Lagrangian wrt the primal variables for inequality
            # constraints is:
            #  grad_objective + grad_constraint * relu(multiplier + penalty_coeff * violation)
            violation = self._manual_violation(manual_x)[features]
            aux_grad = manual_multiplier[features] + manual_penalty_coeff[features] * violation
            if self.is_inequality:
                aux_grad.relu_()
            constraint_grad = lhs[features].t() @ aux_grad
        else:
            constraint_grad = lhs[features].t() @ manual_multiplier[features]
        manual_x_grad = obj_grad + constraint_grad

        return manual_x - PRIMAL_LR * manual_x_grad

    def _manual_dual_step(self, manual_x, manual_multiplier, strict_features, manual_penalty_coeff=None):
        violation = self._manual_violation(manual_x, strict=True)[strict_features]
        if manual_penalty_coeff is None:
            manual_multiplier[strict_features] = manual_multiplier[strict_features] + DUAL_LR * violation
        else:
            manual_multiplier[strict_features] = (
                manual_multiplier[strict_features] + manual_penalty_coeff[strict_features] * violation
            )

        if self.is_inequality:
            manual_multiplier.relu_()

        return manual_multiplier

    def _manual_primal_lagrangian(
        self, manual_x, manual_multiplier, features, manual_penalty_coeff: Optional[torch.Tensor] = None
    ):
        loss = torch.sum(manual_x**2)
        violation = self._manual_violation(manual_x)[features]

        if manual_penalty_coeff is None:
            return loss + torch.dot(manual_multiplier[features], violation)

        penalty_term = manual_multiplier[features] + manual_penalty_coeff[features] * violation
        if self.is_inequality:
            penalty_term.relu_()

        aug_lag = loss + 0.5 * torch.dot(
            1 / manual_penalty_coeff[features], penalty_term**2 - manual_multiplier[features] ** 2
        )
        return aug_lag

    def _manual_dual_lagrangian(self, manual_x, manual_multiplier, strict_features, manual_penalty_coeff=None):
        violation = self._manual_violation(manual_x, strict=True)[strict_features]
        if manual_penalty_coeff is None:
            return torch.sum(manual_multiplier[strict_features] * violation)

        # When the penalty coefficient is provided, the dual lagrangian is weighted by
        # both the value of the multiplier and the penalty coefficient.
        # This way, the gradient with respect to the multiplier is the constraint violation times
        # the penalty coefficient, as required by the updates of the Augmented Lagrangian Method.
        return torch.sum(manual_penalty_coeff[strict_features] * manual_multiplier[strict_features] * violation)

    def _manual_update_penalty_coefficients(
        self, manual_x, manual_x_prev, strict_features, alternation_type, manual_penalty_coeff
    ):
        if alternation_type == cooper_test_utils.AlternationType.PRIMAL_DUAL:
            strict_violation = self._manual_violation(manual_x, strict=True)[strict_features]
        else:
            strict_violation = self._manual_violation(manual_x_prev, strict=True)[strict_features]
        if self.is_inequality:
            strict_violation.relu_()
        violated_indices = strict_violation.abs() > PENALTY_VIOLATION_TOLERANCE
        # Update the penalty coefficients for the violated constraints
        manual_penalty_coeff[strict_features[violated_indices]] *= PENALTY_GROWTH_FACTOR

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
        # Dual step
        manual_dual_lagrangian = self._manual_dual_lagrangian(
            manual_x, manual_multiplier, strict_features, manual_penalty_coeff
        )
        manual_multiplier = self._manual_dual_step(manual_x, manual_multiplier, strict_features, manual_penalty_coeff)

        # Primal step
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
        # Primal step
        manual_primal_lagrangian = self._manual_primal_lagrangian(
            manual_x, manual_multiplier, features, manual_penalty_coeff
        )
        manual_observed_multipliers = manual_multiplier.clone()
        manual_x = self._manual_primal_step(manual_x, manual_multiplier, features, manual_penalty_coeff)

        # Dual step
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
        manual_x_copy = manual_x.clone()  # x_t
        manual_multiplier_copy = manual_multiplier.clone()

        # Extrapolation step
        manual_x = self._manual_primal_step(manual_x_copy, manual_multiplier_copy, features)  # x_{t+1/2}
        manual_multiplier = self._manual_dual_step(manual_x_copy, manual_multiplier_copy.clone(), strict_features)

        # Update step
        manual_primal_lagrangian = self._manual_primal_lagrangian(manual_x, manual_multiplier, features)
        manual_observed_multipliers = manual_multiplier.clone()
        manual_dual_lagrangian = self._manual_dual_lagrangian(manual_x, manual_multiplier, strict_features)

        lhs = self.lhs_sur if self.use_surrogate else self.lhs
        manual_x_grad = 2 * manual_x + lhs[features].t() @ manual_multiplier[features]
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
