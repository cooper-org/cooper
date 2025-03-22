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

        self.expects_multiplier = formulation_type.expects_multiplier
        self.expects_penalty_coefficient = formulation_type.expects_penalty_coefficient
        self.use_penalty_updater = formulation_type != cooper.formulations.Lagrangian

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
        x = torch.nn.Parameter(torch.ones(self.num_variables, device=self.device))
        optimizer_class = cooper.optim.ExtraSGD if extrapolation else torch.optim.SGD
        primal_optimizers = optimizer_class([x], lr=PRIMAL_LR)

        cooper_optimizer = testing.build_cooper_optimizer(
            cmp=self.cmp,
            primal_optimizers=primal_optimizers,
            extrapolation=extrapolation,
            dual_optimizer_class=optimizer_class,
            alternation_type=alternation_type,
            dual_optimizer_kwargs={"lr": DUAL_LR},
        )

        penalty_updater = None
        if self.use_penalty_updater:
            penalty_updater = cooper.penalty_coefficients.MultiplicativePenaltyCoefficientUpdater(
                growth_factor=PENALTY_GROWTH_FACTOR, violation_tolerance=PENALTY_VIOLATION_TOLERANCE, has_restart=False
            )

        manual_x = torch.ones(self.num_variables, device=self.device)
        manual_multiplier = None
        if self.expects_multiplier:
            manual_multiplier = torch.zeros(self.num_constraints, device=self.device)
        manual_penalty_coeff = None
        if self.expects_penalty_coefficient:
            manual_penalty_coeff = torch.ones(self.num_constraints, device=self.device)

        # ----------------------- Iterations -----------------------
        for step in range(2):
            roll_kwargs = {"compute_cmp_state_kwargs": {"x": x, "seed": step}}
            if alternation_type == testing.AlternationType.PRIMAL_DUAL:
                roll_kwargs["compute_violations_kwargs"] = {"x": x, "seed": step}
            roll_out = cooper_optimizer.roll(**roll_kwargs)
            if self.use_penalty_updater:
                penalty_updater.step(roll_out.cmp_state.observed_constraints)

            primal_observed_multipliers = None
            dual_observed_multipliers = None
            if self.expects_multiplier:
                primal_observed_multipliers = torch.cat(
                    list(roll_out.primal_lagrangian_store.observed_multiplier_values())
                )
                dual_observed_multipliers = torch.cat(list(roll_out.dual_lagrangian_store.observed_multiplier_values()))

            # The CMP has only one constraint, so we can use the first element
            features = next(iter(roll_out.cmp_state.named_observed_constraint_features()))[1]
            if features is None:
                features = torch.arange(self.num_constraints, device=self.device, dtype=torch.long)

            strict_features = next(iter(roll_out.cmp_state.named_observed_strict_constraint_features()))[1]
            if strict_features is None:
                strict_features = features

            manual_x_prev = manual_x.clone()
            # Manual step
            (
                manual_x,
                manual_multiplier,
                manual_primal_lagrangian,
                manual_dual_lagrangian,
                manual_primal_observed_multipliers,
                manual_dual_observed_multipliers,
            ) = self.manual_roll(
                manual_x,
                manual_multiplier,
                features,
                strict_features,
                alternation_type,
                manual_penalty_coeff,
                extrapolation,
            )

            if self.use_penalty_updater:
                # Update penalty coefficients for the Quadratic Penalty formulation
                self._update_penalty_coefficients(
                    manual_x, manual_x_prev, strict_features, alternation_type, manual_penalty_coeff
                )

            # Check manual and cooper outputs are close
            assert torch.allclose(x, manual_x)
            assert torch.allclose(roll_out.primal_lagrangian_store.lagrangian, manual_primal_lagrangian)
            if self.expects_multiplier:
                assert torch.allclose(primal_observed_multipliers, manual_primal_observed_multipliers[features])
                assert torch.allclose(dual_observed_multipliers, manual_dual_observed_multipliers[strict_features])
                assert torch.allclose(roll_out.dual_lagrangian_store.lagrangian, manual_dual_lagrangian)

    def _violation(self, x, strict=False):
        """Compute the constraint violations given the primal variables.
        If strict is True, the strict violations are computed.
        Otherwise, the surrogate violations are computed if the surrogates are provided.
        """
        if not strict and self.use_surrogate:
            return self.lhs_sur @ x - self.rhs
        return self.lhs @ x - self.rhs

    def _primal_gradient(self, x, multiplier, features, penalty_coeff):
        lhs = self.lhs_sur if self.use_surrogate else self.lhs
        obj_grad = 2 * x
        if self.expects_penalty_coefficient:
            # The gradient of the Augmented Lagrangian wrt the primal variables for inequality
            # constraints is: grad_objective + grad_constraint * (multiplier + penalty_coeff * relu(violation))
            violation = self._violation(x)[features]
            if self.is_inequality:
                violation.relu_()
            aux_grad = penalty_coeff[features] * violation
            if self.expects_multiplier:
                aux_grad += multiplier[features]
            constraint_grad = lhs[features].t() @ aux_grad
        else:
            constraint_grad = lhs[features].t() @ multiplier[features]

        return obj_grad + constraint_grad

    def _primal_step(self, x, multiplier, features, penalty_coeff):
        x_grad = self._primal_gradient(x, multiplier, features, penalty_coeff)
        return x - PRIMAL_LR * x_grad

    def _dual_step(self, x, multiplier, strict_features):
        if not self.expects_multiplier:
            # Quadratic Penalty formulation
            return None

        violation = self._violation(x, strict=True)[strict_features]
        multiplier[strict_features] = multiplier[strict_features] + DUAL_LR * violation
        if self.is_inequality:
            multiplier.relu_()

        return multiplier

    def _primal_lagrangian(self, x, multiplier, features, penalty_coeff: Optional[torch.Tensor]):
        loss = torch.sum(x**2)
        violation = self._violation(x)[features]

        if not self.expects_penalty_coefficient:
            # Lagrangian formulation
            return loss + torch.dot(multiplier[features], violation)

        if not self.expects_multiplier:
            # Quadratic Penalty formulation
            if self.is_inequality:
                violation.relu_()
            return loss + 0.5 * torch.dot(penalty_coeff[features], violation**2)

        penalty_term = multiplier[features] + penalty_coeff[features] * violation
        if self.is_inequality:
            penalty_term.relu_()

        aug_lag = loss + 0.5 * torch.dot(1 / penalty_coeff[features], penalty_term**2 - multiplier[features] ** 2)
        return aug_lag

    def _dual_lagrangian(self, x, multiplier, strict_features):
        if not self.expects_multiplier:
            # Quadratic Penalty formulation
            return None

        violation = self._violation(x, strict=True)[strict_features]

        return torch.sum(multiplier[strict_features] * violation)

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

    def _simultaneous_roll(self, x, multiplier, features, strict_features, penalty_coeff):
        primal_lagrangian = self._primal_lagrangian(x, multiplier, features, penalty_coeff)
        observed_multipliers = multiplier.clone() if multiplier is not None else None
        dual_lagrangian = self._dual_lagrangian(x, multiplier, strict_features)

        x, multiplier = (
            self._primal_step(x, multiplier, features, penalty_coeff),
            self._dual_step(x, multiplier, strict_features),
        )

        # primal and dual observed multipliers are the same
        return x, multiplier, primal_lagrangian, dual_lagrangian, observed_multipliers, observed_multipliers

    def _dual_primal_roll(self, x, multiplier, features, strict_features, penalty_coeff):
        # Dual step
        dual_observed_multipliers = multiplier.clone()
        dual_lagrangian = self._dual_lagrangian(x, multiplier, strict_features)
        multiplier = self._dual_step(x, multiplier, strict_features)

        # Primal step
        primal_lagrangian = self._primal_lagrangian(x, multiplier, features, penalty_coeff)
        primal_observed_multipliers = multiplier.clone()
        x = self._primal_step(x, multiplier, features, penalty_coeff)

        return x, multiplier, primal_lagrangian, dual_lagrangian, primal_observed_multipliers, dual_observed_multipliers

    def _primal_dual_roll(self, x, multiplier, features, strict_features, penalty_coeff):
        # Primal step
        primal_lagrangian = self._primal_lagrangian(x, multiplier, features, penalty_coeff)
        primal_observed_multipliers = multiplier.clone()
        x = self._primal_step(x, multiplier, features, penalty_coeff)

        # Dual step
        dual_observed_multipliers = multiplier.clone()
        dual_lagrangian = self._dual_lagrangian(x, multiplier, strict_features)
        multiplier = self._dual_step(x, multiplier, strict_features)

        return x, multiplier, primal_lagrangian, dual_lagrangian, primal_observed_multipliers, dual_observed_multipliers

    def _extragradient_roll(self, x, multiplier, features, strict_features, penalty_coeff):
        x_copy = x.clone()  # x_t
        multiplier_copy = multiplier.clone()

        # Compute the Lagrangians
        primal_lagrangian = self._primal_lagrangian(x, multiplier, features, penalty_coeff)
        observed_multipliers = multiplier.clone()
        dual_lagrangian = self._dual_lagrangian(x, multiplier, strict_features)

        # Extrapolation step
        x = self._primal_step(x_copy, multiplier_copy, features, penalty_coeff)  # x_{t+1/2}
        multiplier = self._dual_step(x_copy, multiplier_copy.clone(), strict_features)

        # Update step
        x_grad = self._primal_gradient(x, multiplier, features, penalty_coeff)
        x, multiplier = x_copy - PRIMAL_LR * x_grad, self._dual_step(x, multiplier_copy, strict_features)

        # primal and dual observed multipliers are the same
        return x, multiplier, primal_lagrangian, dual_lagrangian, observed_multipliers, observed_multipliers

    @torch.inference_mode()
    def manual_roll(self, x, multiplier, features, strict_features, alternation_type, penalty_coeff, extrapolation):
        if extrapolation:
            return self._extragradient_roll(x, multiplier, features, strict_features, penalty_coeff)
        if alternation_type == testing.AlternationType.FALSE:
            return self._simultaneous_roll(x, multiplier, features, strict_features, penalty_coeff)
        if alternation_type == testing.AlternationType.DUAL_PRIMAL:
            return self._dual_primal_roll(x, multiplier, features, strict_features, penalty_coeff)
        if alternation_type == testing.AlternationType.PRIMAL_DUAL:
            return self._primal_dual_roll(x, multiplier, features, strict_features, penalty_coeff)
        raise ValueError(f"Unknown alternation type: {alternation_type}")
