import pytest
import torch

import cooper


@pytest.fixture(
    params=[
        cooper.optim.AlternatingPrimalDualOptimizer,
        cooper.optim.AlternatingDualPrimalOptimizer,
        cooper.optim.SimultaneousOptimizer,
    ]
)
def cooper_optimizer_class(request):
    return request.param


def test_constrained_optimizer_init_fail_no_primal_optimizer(cooper_optimizer_class, cmp_instance):
    with pytest.raises(TypeError, match=r"No primal optimizer\(s\) was provided for building a ConstrainedOptimizer."):
        cooper_optimizer_class(
            cmp=cmp_instance,
            primal_optimizers=None,
            dual_optimizers=torch.optim.SGD(cmp_instance.dual_parameters(), lr=0.1, maximize=True),
        )


def test_constrained_optimizer_init_fail_no_dual_optimizer(cooper_optimizer_class, cmp_instance):
    with pytest.raises(TypeError, match=r"No dual optimizer\(s\) was provided for building a ConstrainedOptimizer."):
        cooper_optimizer_class(
            cmp=cmp_instance,
            primal_optimizers=torch.optim.SGD([torch.ones(1, requires_grad=True)], lr=0.1),
            dual_optimizers=None,
        )


def test_constrained_optimizer_init_fail_dual_optimizer_not_maximize(cooper_optimizer_class, cmp_instance):
    with pytest.raises(ValueError, match=r"Dual optimizers must be set to carry out maximization steps."):
        cooper_optimizer_class(
            cmp=cmp_instance,
            primal_optimizers=torch.optim.SGD([torch.ones(1, requires_grad=True)], lr=0.1),
            dual_optimizers=torch.optim.SGD(cmp_instance.dual_parameters(), lr=0.1, maximize=False),
        )


def test_alternating_primal_dual_optimizer_roll_fail_compute_violations_has_loss(cmp_instance):
    optimizer = cooper.optim.AlternatingPrimalDualOptimizer(
        cmp=cmp_instance,
        primal_optimizers=torch.optim.SGD([torch.ones(1, requires_grad=True)], lr=0.1),
        dual_optimizers=torch.optim.SGD(cmp_instance.dual_parameters(), lr=0.1, maximize=True),
    )

    def compute_violations():
        return cooper.CMPState(
            loss=torch.tensor(1.0),
            observed_constraints={cmp_instance.eq_constraint: cooper.ConstraintState(violation=torch.tensor(1.0))},
        )

    cmp_instance.compute_violations = compute_violations

    with pytest.raises(
        RuntimeError,
        match=r"Expected `compute_violations` to not populate the loss. "
        r"Please provide this value for the `compute_cmp_state` instead.",
    ):
        optimizer.roll()


def test_alternating_primal_dual_optimizer_roll_no_compute_violations(cmp_instance):
    optimizer = cooper.optim.AlternatingPrimalDualOptimizer(
        cmp=cmp_instance,
        primal_optimizers=torch.optim.SGD([torch.ones(1, requires_grad=True)], lr=0.1),
        dual_optimizers=torch.optim.SGD(cmp_instance.dual_parameters(), lr=0.1, maximize=True),
    )

    optimizer.roll()  # This shouldn't raise any errors


def test_extrapolation_init_fail_no_extrapolation_method(cmp_instance):
    with pytest.raises(RuntimeError, match=r"Some of the provided optimizers do not have an extrapolation method."):
        cooper.optim.ExtrapolationConstrainedOptimizer(
            cmp=cmp_instance,
            primal_optimizers=torch.optim.SGD([torch.ones(1, requires_grad=True)], lr=0.1),
            dual_optimizers=torch.optim.SGD(cmp_instance.dual_parameters(), lr=0.1, maximize=True),
        )
