import math
from typing import Optional

import pytest
import torch

import cooper
import testing

PRIMAL_LR = 3e-2
DUAL_LR = 2e-1
PENALTY_GROWTH_FACTOR = 1.0 + 2.5e-4
PENALTY_VIOLATION_TOLERANCE = 1e-4


class TestConvergence:
    @pytest.fixture(autouse=True)
    def _setup_cmp(
        self,
        cmp,
        constraint_params,
        constraint_type,
        use_surrogate,
        multiplier_type,
        formulation_type,
        num_constraints,
        num_variables,
        device,
    ):
        self.is_inequality = constraint_type == cooper.ConstraintType.INEQUALITY

        self.cmp = cmp
        self.lhs, self.rhs = constraint_params
        self.lhs_sur = self.cmp.A_sur if self.is_inequality else self.cmp.C_sur

        self.constraint_type = constraint_type
        self.use_surrogate = use_surrogate
        self.is_augmented_lagrangian = formulation_type == cooper.formulations.AugmentedLagrangian
        self.is_indexed_multiplier = multiplier_type == cooper.multipliers.IndexedMultiplier
        self.num_variables = num_variables
        self.num_constraints = num_constraints
        self.device = device
        self.primal_lr = PRIMAL_LR
        self.dual_lr = DUAL_LR / math.sqrt(num_variables)

    def test_manual_step(self, extrapolation, alternation_type):
        """Test the cooper optimizer roll methods implementation.

        This method tests the cooper optimizers by comparing the results with the manual implementation.
        The manual implementation assumes Stochastic Gradient Descent (SGD) is used for both the primal
        and dual optimizers.
        """
        if alternation_type == testing.AlternationType.PRIMAL_DUAL and self.is_indexed_multiplier:
            pytest.skip("Cannot test IndexedMultiplier with PRIMAL_DUAL alternation.")

        x = torch.nn.Parameter(torch.ones(self.num_variables, device=self.device))
        optimizer_class = cooper.optim.ExtraSGD if extrapolation else torch.optim.SGD
        primal_optimizers = optimizer_class([x], lr=PRIMAL_LR)

        cooper_optimizer = testing.build_cooper_optimizer(
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
            penalty_updater = cooper.multipliers.MultiplicativePenaltyCoefficientUpdater(
                growth_factor=PENALTY_GROWTH_FACTOR, violation_tolerance=PENALTY_VIOLATION_TOLERANCE
            )

        roll_kwargs = {"compute_cmp_state_kwargs": {"x": x}}
        if alternation_type == testing.AlternationType.PRIMAL_DUAL:
            roll_kwargs["compute_violations_kwargs"] = {"x": x}

        manual_x = torch.ones(self.num_variables, device=self.device)
        manual_multiplier = torch.zeros(self.num_constraints, device=self.device)
        manual_penalty_coeff = None
        if self.is_augmented_lagrangian:
            manual_penalty_coeff = torch.ones(self.num_constraints, device=self.device)

        # ----------------------- Iterations -----------------------
        for _ in range(2):
            roll_out = cooper_optimizer.roll(**roll_kwargs)
            if self.is_augmented_lagrangian:
                penalty_updater.step(roll_out.cmp_state.observed_constraints)

            if alternation_type == testing.AlternationType.PRIMAL_DUAL:
                observed_multipliers = torch.cat(list(roll_out.dual_lagrangian_store.observed_multiplier_values()))
            else:
                observed_multipliers = torch.cat(list(roll_out.primal_lagrangian_store.observed_multiplier_values()))

            features = next(iter(roll_out.cmp_state.observed_constraint_features()))
            if features is None:
                features = torch.arange(self.num_constraints, device=self.device, dtype=torch.long)

            strict_features = next(iter(roll_out.cmp_state.observed_strict_constraint_features()))
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

            if self.is_augmented_lagrangian:
                # Update penalty coefficients for Augmented Lagrangian Method
                self._update_penalty_coefficients(
                    manual_x, manual_x_prev, strict_features, alternation_type, manual_penalty_coeff
                )

            # Check manual and cooper outputs are close
            assert torch.allclose(observed_multipliers, manual_observed_multipliers[features])
            assert torch.allclose(x, manual_x)
            assert torch.allclose(roll_out.primal_lagrangian_store.lagrangian, manual_primal_lagrangian)
            assert torch.allclose(roll_out.dual_lagrangian_store.lagrangian, manual_dual_lagrangian)

    def _violation(self, x, strict=False):
        """Compute the constraint violations given the primal variables.
        If strict is True, the strict violations are computed.
        Otherwise, the surrogate violations are computed if the surrogates are provided.
        """
        if not strict and self.use_surrogate:
            return self.lhs_sur @ x - self.rhs
        return self.lhs @ x - self.rhs

    def _primal_step(self, x, multiplier, features, penalty_coeff=None):
        lhs = self.lhs_sur if self.use_surrogate else self.lhs
        obj_grad = 2 * x
        if penalty_coeff is not None:
            # The gradient of the Augmented Lagrangian wrt the primal variables for inequality
            # constraints is: grad_objective + grad_constraint * relu(multiplier + penalty_coeff * violation)
            violation = self._violation(x)[features]
            aux_grad = multiplier[features] + penalty_coeff[features] * violation
            if self.is_inequality:
                aux_grad.relu_()
            constraint_grad = lhs[features].t() @ aux_grad
        else:
            constraint_grad = lhs[features].t() @ multiplier[features]
        x_grad = obj_grad + constraint_grad

        return x - PRIMAL_LR * x_grad

    def _dual_step(self, x, multiplier, strict_features, penalty_coeff=None):
        violation = self._violation(x, strict=True)[strict_features]
        if penalty_coeff is None:
            multiplier[strict_features] = multiplier[strict_features] + DUAL_LR * violation
        else:
            multiplier[strict_features] = multiplier[strict_features] + penalty_coeff[strict_features] * violation

        if self.is_inequality:
            multiplier.relu_()

        return multiplier

    def _primal_lagrangian(self, x, multiplier, features, penalty_coeff: Optional[torch.Tensor] = None):
        loss = torch.sum(x**2)
        violation = self._violation(x)[features]

        if penalty_coeff is None:
            return loss + torch.dot(multiplier[features], violation)

        penalty_term = multiplier[features] + penalty_coeff[features] * violation
        if self.is_inequality:
            penalty_term.relu_()

        aug_lag = loss + 0.5 * torch.dot(1 / penalty_coeff[features], penalty_term**2 - multiplier[features] ** 2)
        return aug_lag

    def _dual_lagrangian(self, x, multiplier, strict_features, penalty_coeff=None):
        violation = self._violation(x, strict=True)[strict_features]
        if penalty_coeff is None:
            return torch.sum(multiplier[strict_features] * violation)

        # When the penalty coefficient is provided, the dual lagrangian is weighted by
        # both the value of the multiplier and the penalty coefficient.
        # This way, the gradient with respect to the multiplier is the constraint violation times
        # the penalty coefficient, as required by the updates of the Augmented Lagrangian Method.
        return torch.sum(penalty_coeff[strict_features] * multiplier[strict_features] * violation)

    def _update_penalty_coefficients(self, x, x_prev, strict_features, alternation_type, penalty_coeff):
        if alternation_type == testing.AlternationType.PRIMAL_DUAL:
            strict_violation = self._violation(x, strict=True)[strict_features]
        else:
            strict_violation = self._violation(x_prev, strict=True)[strict_features]
        if self.is_inequality:
            strict_violation.relu_()
        violated_indices = strict_violation.abs() > PENALTY_VIOLATION_TOLERANCE
        # Update the penalty coefficients for the violated constraints
        penalty_coeff[strict_features[violated_indices]] *= PENALTY_GROWTH_FACTOR

    def _simultaneous_roll(self, x, multiplier, features, strict_features):
        primal_lagrangian = self._primal_lagrangian(x, multiplier, features)
        observed_multipliers = multiplier.clone()
        dual_lagrangian = self._dual_lagrangian(x, multiplier, strict_features)

        x, multiplier = self._primal_step(x, multiplier, features), self._dual_step(x, multiplier, strict_features)

        return x, multiplier, primal_lagrangian, dual_lagrangian, observed_multipliers

    def _dual_primal_roll(self, x, multiplier, features, strict_features, penalty_coeff):
        # Dual step
        dual_lagrangian = self._dual_lagrangian(x, multiplier, strict_features, penalty_coeff)
        multiplier = self._dual_step(x, multiplier, strict_features, penalty_coeff)

        # Primal step
        primal_lagrangian = self._primal_lagrangian(x, multiplier, features, penalty_coeff)
        observed_multipliers = multiplier.clone()
        x = self._primal_step(x, multiplier, features, penalty_coeff)

        return x, multiplier, primal_lagrangian, dual_lagrangian, observed_multipliers

    def _primal_dual_roll(self, x, multiplier, features, strict_features, penalty_coeff):
        # Primal step
        primal_lagrangian = self._primal_lagrangian(x, multiplier, features, penalty_coeff)
        observed_multipliers = multiplier.clone()
        x = self._primal_step(x, multiplier, features, penalty_coeff)

        # Dual step
        dual_lagrangian = self._dual_lagrangian(x, multiplier, strict_features, penalty_coeff)
        multiplier = self._dual_step(x, multiplier, strict_features, penalty_coeff)

        return x, multiplier, primal_lagrangian, dual_lagrangian, observed_multipliers

    def _extragradient_roll(self, x, multiplier, features, strict_features):
        x_copy = x.clone()  # x_t
        multiplier_copy = multiplier.clone()

        # Extrapolation step
        x = self._primal_step(x_copy, multiplier_copy, features)  # x_{t+1/2}
        multiplier = self._dual_step(x_copy, multiplier_copy.clone(), strict_features)

        # Update step
        primal_lagrangian = self._primal_lagrangian(x, multiplier, features)
        observed_multipliers = multiplier.clone()
        dual_lagrangian = self._dual_lagrangian(x, multiplier, strict_features)

        lhs = self.lhs_sur if self.use_surrogate else self.lhs
        x_grad = 2 * x + lhs[features].t() @ multiplier[features]
        x, multiplier = x_copy - PRIMAL_LR * x_grad, self._dual_step(x, multiplier_copy, strict_features)

        return x, multiplier, primal_lagrangian, dual_lagrangian, observed_multipliers

    @torch.inference_mode()
    def manual_roll(self, x, multiplier, features, strict_features, alternation_type, penalty_coeff, extrapolation):
        if extrapolation:
            return self._extragradient_roll(x, multiplier, features, strict_features)
        if alternation_type == testing.AlternationType.FALSE:
            return self._simultaneous_roll(x, multiplier, features, strict_features)
        if alternation_type == testing.AlternationType.DUAL_PRIMAL:
            return self._dual_primal_roll(x, multiplier, features, strict_features, penalty_coeff)
        if alternation_type == testing.AlternationType.PRIMAL_DUAL:
            return self._primal_dual_roll(x, multiplier, features, strict_features, penalty_coeff)
        raise ValueError(f"Unknown alternation type: {alternation_type}")
