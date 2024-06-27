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


@pytest.fixture
def cooper_optimizer_no_constraint(cmp_no_constraint, params):
    primal_optimizers = cooper_test_utils.build_primal_optimizers(
        params, primal_optimizer_kwargs=[{"lr": PRIMAL_LR} for _ in range(len(params))]
    )
    cooper_optimizer = cooper_test_utils.build_cooper_optimizer(
        cmp=cmp_no_constraint, primal_optimizers=primal_optimizers
    )
    return cooper_optimizer


def test_convergence_no_constraint(cmp_no_constraint, params, cooper_optimizer_no_constraint):
    for _ in range(2000):
        cooper_optimizer_no_constraint.roll(compute_cmp_state_kwargs=dict(x=torch.cat(params)))

    # Compute the exact solution
    x_star, _ = cmp_no_constraint.compute_exact_solution()

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
