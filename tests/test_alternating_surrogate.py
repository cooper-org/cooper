import pytest
import torch

from tests.helpers import cooper_test_utils, testing_utils

USE_CONSTRAINT_SURROGATE = True


@pytest.mark.parametrize("use_violation_fn", [True, False])
def test_manual_PrimalDual_surrogate(use_violation_fn, Toy2dCMP_problem_properties, Toy2dCMP_params_init, device):
    """Test first two iterations of PrimalDual alternating GDA updates on a toy 2D
    problem with surrogate constraints."""

    use_ineq_constraints = Toy2dCMP_problem_properties["use_ineq_constraints"]
    if not use_ineq_constraints:
        pytest.skip("Alternating updates requires a problem with constraints.")

    if not torch.allclose(Toy2dCMP_params_init, torch.tensor([0.0, -1.0], device=device)):
        pytest.skip("Manual alternating test only considers the case of initialization at [0, -1]")

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=False, params_init=Toy2dCMP_params_init
    )

    # Only performing this test for the case of a single primal optimizer
    assert isinstance(params, torch.nn.Parameter)

    cmp = cooper_test_utils.Toy2dCMP(
        use_ineq_constraints=True, use_constraint_surrogate=USE_CONSTRAINT_SURROGATE, device=device
    )

    mktensor = testing_utils.mktensor(device=device)

    PRIMAL_LR = 1e-2
    DUAL_LR = 1e-2

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
        primal_optimizers=primal_optimizers,
        cmp=cmp,
        extrapolation=False,
        alternation_type=cooper_test_utils.AlternationType.PRIMAL_DUAL,
        dual_optimizer_class=torch.optim.SGD,
        dual_optimizer_kwargs={"lr": DUAL_LR},
    )

    roll_kwargs = {
        "compute_cmp_state_kwargs": dict(params=params),
        "compute_violations_kwargs": dict(params=params) if use_violation_fn else dict(params=None),
    }

    x0_y0 = mktensor([0.0, -1.0])
    lmbda0 = mktensor([0.0, 0.0])

    # ----------------------- First iteration -----------------------
    # The returned CMPState when using PrimalDual updates is measured _after_ performing
    # the primal update.
    loss, cmp_state, primal_ls, dual_ls = cooper_optimizer.roll(**roll_kwargs)
    _cmp_state = cmp.compute_cmp_state(x0_y0)

    # No dual update yet, so the observed multipliers should be zero, matching lmdba0
    observed_multipliers = torch.cat(list(primal_ls.observed_multiplier_values()))
    assert torch.allclose(observed_multipliers, lmbda0)

    # analytical_gradients computes the gradients of the loss and surrogate constraints
    grads_x0_y0 = cmp.analytical_gradients(x0_y0)
    x1_y1 = x0_y0 - PRIMAL_LR * (grads_x0_y0[0] + torch.sum(lmbda0 * grads_x0_y0[1]))
    assert torch.allclose(params, x1_y1)

    # The value of the primal Lagrangian should be the loss at (x0, y0) plus the
    # constraint contributions _before_ the primal update
    violations_before_primal_update = mktensor(list(_cmp_state.observed_violations()))
    assert torch.allclose(primal_ls.lagrangian, _cmp_state.loss + torch.sum(violations_before_primal_update * lmbda0))

    strict_violations_after_primal_update = mktensor(list(cmp_state.observed_strict_violations()))
    # The value of the dual Lagrangian should only count the constraint contributions
    # after the primal update, using the original multipliers
    assert torch.allclose(dual_ls.lagrangian, torch.sum(strict_violations_after_primal_update * lmbda0))

    # Lambda update uses the violations after the primal update.
    # Note that the dual Lagrangian store does NOT include the updated multipliers
    # since the update happens inside roll _after_ computing the Lagrangian store.
    lmbda1 = torch.relu(lmbda0 + DUAL_LR * strict_violations_after_primal_update)

    # ----------------------- Second iteration -----------------------
    loss, cmp_state, primal_ls, dual_ls = cooper_optimizer.roll(**roll_kwargs)
    _cmp_state = cmp.compute_cmp_state(x1_y1)

    # At this stage we have carried out [primal_update, dual_update, primal_update], so
    # we can verify that the observed multipliers have the right values after their
    # update in the _previous_ step.
    observed_multipliers = torch.cat(list(primal_ls.observed_multiplier_values()))
    assert torch.allclose(observed_multipliers, lmbda1)

    # analytical_gradients computes the gradients of the loss and surrogate constraints
    grads_x1_y1 = cmp.analytical_gradients(x1_y1)
    x2_y2 = x1_y1 - PRIMAL_LR * (grads_x1_y1[0] + torch.sum(lmbda1 * grads_x1_y1[1]))

    # NOTE: this test requires a relaxed tolerance of 1e-4. investigate why?
    assert torch.allclose(params, x2_y2, atol=1e-4)

    # The value of the primal Lagrangian should be the loss at (x1, y1) plus the
    # constraint contributions _before_ the primal update
    violations_before_primal_update = mktensor(list(_cmp_state.observed_violations()))
    assert torch.allclose(primal_ls.lagrangian, _cmp_state.loss + torch.sum(violations_before_primal_update * lmbda1))

    strict_violations_after_primal_update = mktensor(list(cmp_state.observed_strict_violations()))
    # The value of the dual Lagrangian should only count the constraint contributions
    # after the primal update
    assert torch.allclose(dual_ls.lagrangian, torch.sum(strict_violations_after_primal_update * lmbda1))


def test_manual_DualPrimal_surrogate(Toy2dCMP_problem_properties, Toy2dCMP_params_init, device):
    """Test first two iterations of DualPrimal alternating GDA updates on a toy 2D
    problem with surrogate constraints."""

    use_ineq_constraints = Toy2dCMP_problem_properties["use_ineq_constraints"]
    if not use_ineq_constraints:
        pytest.skip("Alternating updates requires a problem with constraints.")

    if not torch.allclose(Toy2dCMP_params_init, torch.tensor([0.0, -1.0], device=device)):
        pytest.skip("Manual alternating test only considers the case of initialization at [0, -1]")

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=False, params_init=Toy2dCMP_params_init
    )

    # Only performing this test for the case of a single primal optimizer
    assert isinstance(params, torch.nn.Parameter)

    cmp = cooper_test_utils.Toy2dCMP(
        use_ineq_constraints=True, use_constraint_surrogate=USE_CONSTRAINT_SURROGATE, device=device
    )

    mktensor = testing_utils.mktensor(device=device)

    PRIMAL_LR = 1e-2
    DUAL_LR = 1e-2

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
        primal_optimizers=primal_optimizers,
        cmp=cmp,
        extrapolation=False,
        alternation_type=cooper_test_utils.AlternationType.DUAL_PRIMAL,
        dual_optimizer_class=torch.optim.SGD,
        dual_optimizer_kwargs={"lr": DUAL_LR},
    )

    roll_kwargs = {"compute_cmp_state_kwargs": dict(params=params)}

    x0_y0 = mktensor([0.0, -1.0])
    lmbda0 = mktensor([0.0, 0.0])

    # ----------------------- First iteration -----------------------
    # The CMPState returned when using DualPrimal is measured _before any_ updates
    loss, cmp_state, primal_ls, dual_ls = cooper_optimizer.roll(**roll_kwargs)
    _cmp_state = cmp.compute_cmp_state(x0_y0)

    strict_violations_before_primal_update = mktensor(list(cmp_state.observed_strict_violations()))
    lmbda1 = torch.relu(lmbda0 + DUAL_LR * strict_violations_before_primal_update)

    # The primal Lagrangian store uses the multipliers _after_ the dual update
    observed_multipliers = torch.cat(list(primal_ls.observed_multiplier_values()))
    assert torch.allclose(observed_multipliers, lmbda1)

    # analytical_gradients computes the gradients of the loss and surrogate constraints
    grads_x0_y0 = cmp.analytical_gradients(x0_y0)
    # Need to use the updated multipliers when computing the primal update
    x1_y1 = x0_y0 - PRIMAL_LR * (grads_x0_y0[0] + torch.sum(lmbda1 * grads_x0_y0[1]))
    # TODO(galllego-posada): check if need for relaxed tolerance comes from using surrogate constraints?
    assert torch.allclose(params, x1_y1, atol=1e-4)

    # The value of the dual Lagrangian should only count the constraint contributions
    # before the primal update, using the original multipliers
    assert torch.allclose(dual_ls.lagrangian, torch.sum(strict_violations_before_primal_update * lmbda0))

    # The value of the primal Lagrangian should be the loss at (x0, y0) plus the
    # constraint contributions _before_ the primal update; but using the updated
    # multipliers
    violations_before_primal_update = mktensor(list(_cmp_state.observed_violations()))
    assert torch.allclose(primal_ls.lagrangian, _cmp_state.loss + torch.sum(violations_before_primal_update * lmbda1))

    # ----------------------- Second iteration -----------------------
    loss, cmp_state, primal_ls, dual_ls = cooper_optimizer.roll(**roll_kwargs)
    _cmp_state = cmp.compute_cmp_state(x1_y1)

    strict_violations_before_primal_update = mktensor(list(cmp_state.observed_strict_violations()))
    lmbda2 = torch.relu(lmbda1 + DUAL_LR * strict_violations_before_primal_update)

    # The primal Lagrangian store uses the multipliers _after_ the dual update
    observed_multipliers = torch.cat(list(primal_ls.observed_multiplier_values()))
    assert torch.allclose(observed_multipliers, lmbda2)

    # analytical_gradients computes the gradients of the loss and surrogate constraints
    grads_x1_y1 = cmp.analytical_gradients(x1_y1)
    # Need to use the updated multipliers when computing the primal update
    x2_y2 = x1_y1 - PRIMAL_LR * (grads_x1_y1[0] + torch.sum(lmbda2 * grads_x1_y1[1]))
    # TODO(galllego-posada): check if need for relaxed tolerance comes from using surrogate constraints?
    assert torch.allclose(params, x2_y2, atol=1e-4)

    # The value of the dual Lagrangian should only count the constraint contributions
    # before the second primal update, using multipliers before their second update
    assert torch.allclose(dual_ls.lagrangian, torch.sum(strict_violations_before_primal_update * lmbda1))

    # The value of the primal Lagrangian should be the loss at (x1, y1) plus the
    # constraint contributions _before_ the primal update; but using the updated
    # multipliers
    violations_before_primal_update = mktensor(list(_cmp_state.observed_violations()))
    target_primal_lagrangian = _cmp_state.loss + torch.sum(violations_before_primal_update * lmbda2)
    # TODO(galllego-posada): check if need for relaxed tolerance comes from using surrogate constraints?
    assert torch.allclose(primal_ls.lagrangian, target_primal_lagrangian, atol=1e-4)


@pytest.mark.parametrize(
    "alternation_type", [cooper_test_utils.AlternationType.PRIMAL_DUAL, cooper_test_utils.AlternationType.DUAL_PRIMAL]
)
@pytest.mark.parametrize("use_defect_fn", [True, False])
def test_convergence_surrogate(alternation_type, use_defect_fn, Toy2dCMP_problem_properties, device):
    pass
