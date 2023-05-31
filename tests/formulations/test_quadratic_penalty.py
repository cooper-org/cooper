import os
import tempfile

import cooper_test_utils
import pytest
import testing_utils
import torch

import cooper


def setup_objects(primal_optimizers, alternating, device):
    const1_penalty_coefficient = cooper.multipliers.PenaltyCoefficient(torch.tensor(1.0, device=device))
    const2_penalty_coefficient = cooper.multipliers.PenaltyCoefficient(torch.tensor(1.0, device=device))

    cmp = cooper_test_utils.Toy2dCMP(
        use_ineq_constraints=True,
        formulation_type=cooper.FormulationType.QUADRATIC_PENALTY,
        penalty_coefficients=[const1_penalty_coefficient, const2_penalty_coefficient],
        device=device,
    )

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
        primal_optimizers=primal_optimizers,
        constraint_groups=[],
        extrapolation=False,
        alternating=alternating,
    )

    return cmp, cooper_optimizer, const1_penalty_coefficient, const2_penalty_coefficient


def test_manual_quadratic_penalty(Toy2dCMP_params_init, device):
    """Test first two iterations of PrimalDual alternating GDA updates on toy 2D problem."""

    if not torch.allclose(Toy2dCMP_params_init, torch.tensor([0.0, -1.0], device=device)):
        pytest.skip("Manual alternating test only considers the case of initialization at [0, -1]")

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=False, params_init=Toy2dCMP_params_init
    )

    alternating = cooper.optim.AlternatingType.FALSE

    mktensor = testing_utils.mktensor(device=device)

    cmp, cooper_optimizer, const1_penalty_coefficient, const2_penalty_coefficient = setup_objects(
        primal_optimizers, alternating, device
    )

    roll_kwargs = {"compute_cmp_state_fn": lambda: cmp.compute_cmp_state(params), "return_multipliers": True}

    x0_y0 = mktensor([0.0, -1.0])

    _cmp_state, lagrangian_store = cooper_optimizer.roll(**roll_kwargs)
    violations = mktensor([_[1].violation for _ in _cmp_state.observed_constraints])
    grad_x0_y0 = cmp.analytical_gradients(x0_y0)

    const_contrib0 = (const1_penalty_coefficient() * violations[0].relu()) * grad_x0_y0[1][0]
    const_contrib1 = (const2_penalty_coefficient() * violations[1].relu()) * grad_x0_y0[1][1]
    x1_y1 = x0_y0 - 1e-2 * (grad_x0_y0[0] + const_contrib0 + const_contrib1)

    assert torch.allclose(params, x1_y1)

    # Increase the penalty coefficients
    const1_penalty_coefficient.value = const1_penalty_coefficient.value * 2
    const2_penalty_coefficient.value = const2_penalty_coefficient.value * 2

    # ------------ Second step of simultaneous updates ------------
    _cmp_state, lagrangian_store = cooper_optimizer.roll(**roll_kwargs)
    violations = mktensor([_[1].violation for _ in _cmp_state.observed_constraints])

    grad_x1_y1 = cmp.analytical_gradients(x1_y1)
    const_contrib0 = (const1_penalty_coefficient() * violations[0].relu()) * grad_x1_y1[1][0]
    const_contrib1 = (const2_penalty_coefficient() * violations[1].relu()) * grad_x1_y1[1][1]
    x2_y2 = x1_y1 - 1e-2 * (grad_x1_y1[0] + const_contrib0 + const_contrib1)

    assert torch.allclose(params, x2_y2)


def test_save_and_load_state_dict(Toy2dCMP_params_init, device):
    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=False, params_init=Toy2dCMP_params_init
    )

    alternating = cooper.optim.AlternatingType.FALSE

    cmp, cooper_optimizer, const1_penalty_coefficient, const2_penalty_coefficient = setup_objects(
        primal_optimizers, alternating, device
    )

    roll_kwargs = {"compute_cmp_state_fn": lambda: cmp.compute_cmp_state(params)}

    for _ in range(10):
        cooper_optimizer.roll(**roll_kwargs)
        # Multiply the penalty coefficients by 1.01
        const1_penalty_coefficient.value = const1_penalty_coefficient() * 1.01
        const2_penalty_coefficient.value = const2_penalty_coefficient() * 1.01

    # Generate checkpoints after 10 steps of training
    penalty1_after10 = const1_penalty_coefficient().clone().detach()
    penalty2_after10 = const2_penalty_coefficient().clone().detach()

    with tempfile.TemporaryDirectory() as tmpdirname:
        torch.save(cmp.constraint_groups[0].state_dict(), os.path.join(tmpdirname, "cg0.pt"))
        torch.save(cmp.constraint_groups[1].state_dict(), os.path.join(tmpdirname, "cg1.pt"))

        cg0_state_dict = torch.load(os.path.join(tmpdirname, "cg0.pt"))
        cg1_state_dict = torch.load(os.path.join(tmpdirname, "cg1.pt"))

    # Train for another 10 steps
    for _ in range(10):
        cooper_optimizer.roll(**roll_kwargs)
        const1_penalty_coefficient.value = const1_penalty_coefficient() * 1.01
        const2_penalty_coefficient.value = const2_penalty_coefficient() * 1.01

    # Reload from checkpoint
    new_const1_penalty_coefficient = cooper.multipliers.PenaltyCoefficient(torch.tensor(1.0, device=device))
    new_const2_penalty_coefficient = cooper.multipliers.PenaltyCoefficient(torch.tensor(1.0, device=device))
    new_cmp = cooper_test_utils.Toy2dCMP(
        use_ineq_constraints=True,
        formulation_type=cooper.FormulationType.QUADRATIC_PENALTY,
        penalty_coefficients=[new_const1_penalty_coefficient, new_const2_penalty_coefficient],
        device=device,
    )
    new_cmp.constraint_groups[0].load_state_dict(cg0_state_dict)
    new_cmp.constraint_groups[1].load_state_dict(cg1_state_dict)

    # The loaded penalty coefficients come from 10 steps of training, so they should be
    # different from the current ones
    new_penalty1_value = new_const1_penalty_coefficient().clone().detach()
    new_penalty2_value = new_const2_penalty_coefficient().clone().detach()
    assert not torch.allclose(new_penalty1_value, const1_penalty_coefficient())
    assert not torch.allclose(new_penalty2_value, const2_penalty_coefficient())

    # They should, however, be the same as the ones recorded before the checkpoint
    assert torch.allclose(new_penalty1_value, penalty1_after10)
    assert torch.allclose(new_penalty2_value, penalty2_after10)
