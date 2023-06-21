import cooper_test_utils
import pytest
import testing_utils
import torch

import cooper


def test_manual_quadratic_penalty(Toy2dCMP_params_init, device):
    """Test first two iterations of PrimalDual alternating GDA updates on toy 2D problem."""

    if not torch.allclose(Toy2dCMP_params_init, torch.tensor([0.0, -1.0], device=device)):
        pytest.skip("Manual alternating test only considers the case of initialization at [0, -1]")

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=False, params_init=Toy2dCMP_params_init
    )

    alternating = cooper.optim.AlternatingType.FALSE

    mktensor = testing_utils.mktensor(device=device)

    const1_penalty_coefficient = cooper.multipliers.DensePenaltyCoefficient(torch.tensor(1.0, device=device))
    const2_penalty_coefficient = cooper.multipliers.DensePenaltyCoefficient(torch.tensor(1.0, device=device))

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
