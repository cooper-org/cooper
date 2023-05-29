import cooper_test_utils
import pytest
import testing_utils
import torch

import cooper


def setup_augmented_lagrangian_objects(primal_optimizers, alternating, device):
    const1_penalty_coefficient = cooper.multipliers.PenaltyCoefficient(torch.tensor(1.0, device=device))
    const2_penalty_coefficient = cooper.multipliers.PenaltyCoefficient(torch.tensor(1.0, device=device))

    cmp = cooper_test_utils.Toy2dCMP(
        use_ineq_constraints=True,
        formulation_type=cooper.FormulationType.AUGMENTED_LAGRANGIAN,
        penalty_coefficients=[const1_penalty_coefficient, const2_penalty_coefficient],
        device=device,
    )

    cooper_optimizer = cooper_test_utils.build_cooper_optimizer_for_Toy2dCMP(
        primal_optimizers=primal_optimizers,
        constraint_groups=cmp.constraint_groups,
        extrapolation=False,
        alternating=alternating,
    )

    return cmp, cooper_optimizer, const1_penalty_coefficient, const2_penalty_coefficient


def test_manual_augmented_lagrangian_simultaneous(Toy2dCMP_params_init, device):
    """Test first two iterations of PrimalDual alternating GDA updates on toy 2D problem."""

    if not torch.allclose(Toy2dCMP_params_init, torch.tensor([0.0, -1.0], device=device)):
        pytest.skip("Manual alternating test only considers the case of initialization at [0, -1]")

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=False, params_init=Toy2dCMP_params_init
    )

    alternating = cooper.optim.AlternatingType.FALSE

    mktensor = testing_utils.mktensor(device=device)

    cmp, cooper_optimizer, const1_penalty_coefficient, const2_penalty_coefficient = setup_augmented_lagrangian_objects(
        primal_optimizers, alternating, device
    )

    roll_kwargs = {"compute_cmp_state_fn": lambda: cmp.compute_cmp_state(params), "return_multipliers": True}

    x0_y0 = mktensor([0.0, -1.0])
    lmbda0 = mktensor([0.0, 0.0])

    # ------------ First step of simultaneous updates ------------
    _cmp_state, lagrangian_store = cooper_optimizer.roll(**roll_kwargs)
    violations = mktensor([_[1].violation for _ in _cmp_state.observed_constraints])

    # Observed multipliers from the lagrangian_store should be before the update, thus
    # matching lmbda0
    assert torch.allclose(torch.cat(lagrangian_store.observed_multipliers), lmbda0)

    grad_x0_y0 = cmp.analytical_gradients(x0_y0)
    # The gradient of the Augmented Lagrangian wrt the primal variables is:
    # ...
    # Both constraints in the Toy2dCMP problem are inequality constraints, so relu
    const_contrib0 = (lmbda0[0] + const1_penalty_coefficient() * violations[0].relu()) * grad_x0_y0[1][0]
    const_contrib1 = (lmbda0[1] + const2_penalty_coefficient() * violations[1].relu()) * grad_x0_y0[1][1]
    x1_y1 = x0_y0 - 1e-2 * (grad_x0_y0[0] + const_contrib0 + const_contrib1)

    assert torch.allclose(params, x1_y1)

    # The update to the multipliers has the learning rate and the penalty coefficient
    lmbda_update = torch.stack(
        [violations[0] * const1_penalty_coefficient(), violations[1] * const2_penalty_coefficient()]
    )
    lmbda1 = torch.relu(lmbda0 + 1e-2 * lmbda_update)

    # Increase the penalty coefficients
    # TODO(juan43ramirez): how to do this in a more elegant way? Maybe a method in the
    # ConstraintGroup class?
    # breakpoint()
    # cmp.constraint_groups[0].formulation.penalty_coefficient.weight.data *= 2
    # cmp.constraint_groups[1].formulation.penalty_coefficient.weight.data *= 2
    const1_penalty_coefficient.value = const1_penalty_coefficient() * 2
    const2_penalty_coefficient.value = const2_penalty_coefficient() * 2

    # ------------ Second step of simultaneous updates ------------
    _cmp_state, lagrangian_store = cooper_optimizer.roll(**roll_kwargs)
    violations = mktensor([_[1].violation for _ in _cmp_state.observed_constraints])

    assert torch.allclose(torch.cat(lagrangian_store.observed_multipliers), lmbda1)

    grad_x1_y1 = cmp.analytical_gradients(x1_y1)
    const_contrib0 = (lmbda1[0] + const1_penalty_coefficient() * violations[0].relu()) * grad_x1_y1[1][0]
    const_contrib1 = (lmbda1[1] + const2_penalty_coefficient() * violations[1].relu()) * grad_x1_y1[1][1]
    x2_y2 = x1_y1 - 1e-2 * (grad_x1_y1[0] + const_contrib0 + const_contrib1)

    assert torch.allclose(params, x2_y2)


def test_manual_augmented_lagrangian_primal_dual():
    pass


def test_manual_augmented_lagrangian_dual_primal():
    pass


@pytest.mark.parametrize(
    "alternating_type", [cooper.optim.AlternatingType.PRIMAL_DUAL, cooper.optim.AlternatingType.DUAL_PRIMAL, False]
)
def test_convergence_augmented_lagrangian(
    alternating_type,
    Toy2dCMP_params_init,
    Toy2dCMP_problem_properties,
    use_multiple_primal_optimizers,
    device,
):
    """Test convergence of Augmented Lagrangian updates on toy 2D problem."""

    use_ineq_constraints = Toy2dCMP_problem_properties["use_ineq_constraints"]
    if not use_ineq_constraints:
        pytest.skip("Alternating updates requires a problem with constraints.")

    params, primal_optimizers = cooper_test_utils.build_params_and_primal_optimizers(
        use_multiple_primal_optimizers=use_multiple_primal_optimizers, params_init=Toy2dCMP_params_init
    )

    cmp, cooper_optimizer, const1_penalty_coefficient, const2_penalty_coefficient = setup_augmented_lagrangian_objects(
        primal_optimizers, alternating_type, device
    )

    compute_cmp_state_fn = lambda: cmp.compute_cmp_state(params)
    roll_kwargs = {"compute_cmp_state_fn": compute_cmp_state_fn}

    if alternating_type == cooper.optim.AlternatingType.PRIMAL_DUAL:
        roll_kwargs["compute_violations_fn"] = lambda: cmp.compute_violations(params)

    for step_id in range(1500):
        cooper_optimizer.roll(**roll_kwargs)
        if step_id % 100 == 0:
            # Increase the penalty coefficients
            const1_penalty_coefficient.value = const1_penalty_coefficient() * 1.1
            const2_penalty_coefficient.value = const2_penalty_coefficient() * 1.1

    for param, exact_solution in zip(params, Toy2dCMP_problem_properties["exact_solution"]):
        # NOTE: this test requires a relaxed tolerance of 1e-4
        assert torch.allclose(param, exact_solution, atol=1e-4)
