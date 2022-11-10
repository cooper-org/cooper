#!/usr/bin/env python

"""Tests for Extrapolation optimizers."""

from .helpers import cooper_test_utils
import pytest
import torch


def problem_data(aim_device, alternating):

    test_problem_data = cooper_test_utils.build_test_problem(
        aim_device=aim_device,
        primal_optim_cls=torch.optim.SGD,
        primal_init=[0.0, -1.0],
        dual_optim_cls=torch.optim.SGD,
        use_ineq=True,
        use_proxy_ineq=False,
        use_mult_model=False,
        dual_restarts=False,
        alternating=alternating,
    )

    # params, cmp, coop, formulation, device, mktensor
    return test_problem_data.as_tuple()


@pytest.mark.parametrize("aim_device", ["cpu", "cuda"])
@pytest.mark.parametrize("alternating", [True, False])
@pytest.mark.parametrize("use_defect_fn", [True, False])
def test_manual_alternating(aim_device, alternating, use_defect_fn):
    """
    Test first step of alternating vs non-alternating GDA updates on toy 2D problem.
    """

    params, cmp, coop, formulation, _, mktensor = problem_data(aim_device, alternating)

    defect_fn = cmp.defect_fn if use_defect_fn else None

    coop.zero_grad()
    lagrangian = formulation.compute_lagrangian(cmp.closure, params)

    # Check loss, proxy and non-proxy defects after forward pass
    assert torch.allclose(lagrangian, mktensor(2.0))
    assert torch.allclose(cmp.state.loss, mktensor(2.0))
    assert torch.allclose(cmp.state.ineq_defect, mktensor([2.0, -2.0]))
    assert cmp.state.eq_defect is None

    # Multiplier initialization
    assert torch.allclose(formulation.state()[0], mktensor([0.0, 0.0]))
    assert formulation.state()[1] is None

    # Check primal and dual gradients after backward. Dual gradient must match
    # ineq_defect
    formulation.backward(lagrangian)
    assert torch.allclose(params.grad, mktensor([0.0, -4.0]))
    assert torch.allclose(formulation.state()[0].grad, cmp.state.ineq_defect)

    # Check updated primal and dual variable values
    coop.step(closure=cmp.closure, params=params, defect_fn=defect_fn)

    assert torch.allclose(params, mktensor([0.0, -0.96]))

    if alternating:
        # Loss and violation measurements taken the initialization point [0, -0.96]
        if use_defect_fn:
            # If we use defect_fn, the loss at this point is not recomputed, so we test
            # with the _previous_ loss
            assert torch.allclose(cmp.state.loss, mktensor([2.0]))
        else:
            # If alternating step uses closure, loss is indeed computed at the new primal
            # location.
            assert torch.allclose(cmp.state.loss, mktensor([1.8432]))

        # The constraint defects are [1.96, -1.96]
        assert torch.allclose(formulation.state()[0], mktensor([1.96 * 1e-2, 0.0]))
    else:
        # Loss and violation measurements taken the initialization point [0, -1]
        assert torch.allclose(cmp.state.loss, mktensor([2.0]))
        # In this case the defects are [2., -2.]
        assert torch.allclose(formulation.state()[0], mktensor([2.0 * 1e-2, 0.0]))


@pytest.mark.parametrize("aim_device", ["cpu", "cuda"])
@pytest.mark.parametrize("alternating", [True])
@pytest.mark.parametrize("use_defect_fn", [True, False])
def test_convergence_alternating(aim_device, alternating, use_defect_fn):
    """
    Test convergence of alternating GDA updates on toy 2D problem.
    """

    params, cmp, coop, formulation, _, mktensor = problem_data(aim_device, alternating)

    defect_fn = cmp.defect_fn if use_defect_fn else None

    for step_id in range(1500):
        coop.zero_grad()

        # When using the unconstrained formulation, lagrangian = loss
        lagrangian = formulation.compute_lagrangian(closure=cmp.closure, params=params)
        formulation.backward(lagrangian)

        # Need to pass closure to step function to perform alternating updates
        if use_defect_fn:
            coop.step(defect_fn=defect_fn, params=params)
        else:
            coop.step(closure=cmp.closure, params=params)

    assert torch.allclose(params[0], torch.tensor(2.0 / 3.0))
    assert torch.allclose(params[1], torch.tensor(1.0 / 3.0))
