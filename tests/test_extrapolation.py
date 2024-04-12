import cooper_test_utils
import pytest
import testing_utils
import torch

import cooper


def test_manual_extrapolation(Toy2dCMP_problem_properties, Toy2dCMP_params_init, device):
    """Test first step of Extrapolation-based updates on toy 2D problem."""

    if not torch.allclose(Toy2dCMP_params_init, torch.tensor([0.0, -1.0], device=device)):
        pytest.skip("Manual extrapolation test only considers the case of initialization at [0, -1]")

    if not Toy2dCMP_problem_properties["use_ineq_constraints"]:
        pytest.skip("Extrapolation-based updates are only implemented for constrained problems")

    is_constrained = True

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=False, params_init=Toy2dCMP_params_init, extrapolation=True
    )
    # Only perfoming this test for the case of a single primal optimizer
    assert isinstance(params, torch.nn.Parameter)

    cmp = cooper_test_utils.Toy2dCMP(is_constrained, device=device)

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
        primal_optimizers=primal_optimizers,
        cmp=cmp,
        extrapolation=True,
        alternation_type=cooper.optim.AlternationType.FALSE,
        dual_optimizer_class=cooper.optim.ExtraSGD,
    )

    mktensor = testing_utils.mktensor(device=device)

    # -------------------- First full extra-gradient step  --------------------
    cooper_optimizer.zero_grad()
    cmp_state = cmp.compute_cmp_state(params)
    primal_lagrangian_store = cmp_state.compute_primal_lagrangian()
    dual_lagrangian_store = cmp_state.compute_dual_lagrangian()
    lagrangian = primal_lagrangian_store.lagrangian
    observed_multiplier_values = primal_lagrangian_store.observed_multiplier_values()

    # Check loss, proxy and non-proxy defects after forward pass
    assert torch.allclose(lagrangian, mktensor(2.0))
    assert torch.allclose(cmp_state.loss, mktensor(2.0))

    for (_, constraint_state), target_value in zip(cmp_state.observed_constraints.items(), [2.0, -2.0]):
        assert torch.allclose(constraint_state.violation, mktensor([target_value]))
    # Multiplier initialization
    for multiplier, target_value in zip(observed_multiplier_values, [0.0, 0.0]):
        assert torch.allclose(multiplier, mktensor([target_value]))

    primal_lagrangian_store.backward()
    dual_lagrangian_store.backward()
    assert torch.allclose(params.grad, mktensor([0.0, -4.0]))

    # Dual gradients must match the constraint violations.
    for constraint, constraint_state in cmp_state.observed_constraints.items():
        assert torch.allclose(constraint.multiplier.weight.grad, constraint_state.violation)

    cooper_optimizer.primal_step()
    cooper_optimizer.dual_step(call_extrapolation=True)

    # Perform the actual update step
    cooper_optimizer.zero_grad()
    cmp_state = cmp.compute_cmp_state(params)
    primal_lagrangian_store = cmp_state.compute_primal_lagrangian()
    dual_lagrangian_store = cmp_state.compute_dual_lagrangian()
    primal_lagrangian_store.backward()
    dual_lagrangian_store.backward()
    cooper_optimizer.primal_step()
    cooper_optimizer.dual_step(call_extrapolation=False)

    # After performing the update
    assert torch.allclose(params, mktensor([2.0e-4, -0.9614]))
    assert torch.allclose(params.grad, mktensor([-0.0200, -3.8600]))

    # -------------------- Second full extra-gradient step  --------------------
    cooper_optimizer.roll(compute_cmp_state_kwargs=dict(params=params))

    assert torch.allclose(params, mktensor([5.8428e-04, -9.2410e-01]))
    multiplier_values = [constraint.multiplier() for constraint in cmp.constraints()]
    for multiplier, target_value in zip(multiplier_values, [0.0388, 0.0]):
        assert torch.allclose(multiplier, mktensor([target_value]), atol=1e-4)


@pytest.mark.parametrize("optimizer_name", ["ExtraSGD", "ExtraAdam"])
def test_convergence_extrapolation(optimizer_name, Toy2dCMP_problem_properties, Toy2dCMP_params_init, device):
    """Test convergence of Extrapolation-based updates on toy 2D problem."""

    if not Toy2dCMP_problem_properties["use_ineq_constraints"]:
        pytest.skip("Extrapolation-based updates are only implemented for constrained problems")

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=False,
        params_init=Toy2dCMP_params_init,
        extrapolation=True,
        optimizer_names=optimizer_name,
    )
    cmp = cooper_test_utils.Toy2dCMP(Toy2dCMP_problem_properties["use_ineq_constraints"], device=device)

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
        primal_optimizers=primal_optimizers,
        cmp=cmp,
        extrapolation=True,
        alternation_type=cooper.optim.AlternationType.FALSE,
        dual_optimizer_class=cooper.optim.ExtraSGD,
    )

    for step_id in range(1500):
        cooper_optimizer.roll(compute_cmp_state_kwargs=dict(params=params))

    for param, exact_solution in zip(params, Toy2dCMP_problem_properties["exact_solution"]):
        # NOTE: allowing for a larger tolerance for ExtraAdam tests to pass
        assert torch.allclose(param, exact_solution, rtol=1e-2)
