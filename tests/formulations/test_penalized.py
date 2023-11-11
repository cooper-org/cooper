import cooper_test_utils
import pytest
import testing_utils
import torch

import cooper


def test_manual_penalized(Toy2dCMP_params_init, device):
    """Test first two iterations of PrimalDual alternating GDA updates on toy 2D problem."""

    if not torch.allclose(Toy2dCMP_params_init, torch.tensor([0.0, -1.0], device=device)):
        pytest.skip("Manual alternating test only considers the case of initialization at [0, -1]")

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=False, params_init=Toy2dCMP_params_init
    )

    mktensor = testing_utils.mktensor(device=device)

    penalty_coefficient0 = cooper.multipliers.DensePenaltyCoefficient(torch.tensor(1.0, device=device))
    penalty_coefficient1 = cooper.multipliers.DensePenaltyCoefficient(torch.tensor(1.0, device=device))

    cmp = cooper_test_utils.Toy2dCMP(
        use_ineq_constraints=True,
        formulation_type=cooper.FormulationType.PENALTY,
        penalty_coefficients=(penalty_coefficient0, penalty_coefficient1),
        constraint_type=cooper.ConstraintType.PENALTY,
        device=device,
    )

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
        primal_optimizers=primal_optimizers,
        multipliers=cmp.multipliers,
        extrapolation=False,
        alternation_type=cooper.optim.AlternationType.FALSE,
    )

    roll_kwargs = {"compute_cmp_state_fn": lambda: cmp.compute_cmp_state(params), "return_multipliers": True}

    x0_y0 = mktensor([0.0, -1.0])

    _cmp_state, lagrangian_store = cooper_optimizer.roll(**roll_kwargs)
    grad_x0_y0 = cmp.analytical_gradients(x0_y0)

    const_contrib0 = penalty_coefficient0() * grad_x0_y0[1][0]
    const_contrib1 = penalty_coefficient1() * grad_x0_y0[1][1]
    x1_y1 = x0_y0 - 1e-2 * (grad_x0_y0[0] + const_contrib0 + const_contrib1)

    assert torch.allclose(params, x1_y1)

    # Increase the penalty coefficients
    penalty_coefficient0.value = penalty_coefficient0.value * 2
    penalty_coefficient1.value = penalty_coefficient1.value * 2

    # ------------ Second step of simultaneous updates ------------
    _cmp_state, lagrangian_store = cooper_optimizer.roll(**roll_kwargs)

    grad_x1_y1 = cmp.analytical_gradients(x1_y1)
    const_contrib0 = penalty_coefficient0() * grad_x1_y1[1][0]
    const_contrib1 = penalty_coefficient1() * grad_x1_y1[1][1]
    x2_y2 = x1_y1 - 1e-2 * (grad_x1_y1[0] + const_contrib0 + const_contrib1)

    assert torch.allclose(params, x2_y2)


def test_save_and_load_state_dict():
    # TODO(gallego-posada): Implement test to verify formulation checkpointing works.
    pass
