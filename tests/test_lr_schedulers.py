#!/usr/bin/env python

"""Tests for LR schedulers."""

import cooper_test_utils
import pytest
import torch

import cooper


@pytest.mark.parametrize("aim_device", ["cpu", "cuda"])
@pytest.mark.parametrize("scheduler_name", ["ExponentialLR", "ConstantLR"])
@pytest.mark.parametrize("optimizer_cls", [torch.optim.SGD, cooper.optim.ExtraSGD])
def test_lr_schedulers(aim_device, scheduler_name, optimizer_cls):
    """
    Check behavior of LR schedulers.
    """

    try:
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name)

        if scheduler_name == "ExponentialLR":
            scheduler_kwargs = {"gamma": 0.1}
        elif scheduler_name == "ConstantLR":
            scheduler_kwargs = {"factor": 0.5, "total_iters": 4}

    except:
        scheduler_class, scheduler_kwargs = None, None
        pytest.skip("Requested scheduler is not implemented in this version of Pytorch.")

    base_lr = 1e1

    test_problem_data = cooper_test_utils.build_test_problem(
        aim_device=aim_device,
        primal_optim_cls=optimizer_cls,
        primal_init=[0.0, -1.0],
        dual_optim_cls=optimizer_cls,
        use_ineq=True,
        use_proxy_ineq=True,
        dual_restarts=False,
        alternating=False,
        primal_optim_kwargs={"lr": base_lr},
        dual_optim_kwargs={"lr": base_lr},
        dual_scheduler=cooper.optim.partial_scheduler(scheduler_class, **scheduler_kwargs),
    )

    params, cmp, coop, formulation, _, _ = test_problem_data.as_tuple()

    # Only considering one primal_optimizer on this test.
    primal_optimizer = coop.primal_optimizers[0]
    primal_scheduler = scheduler_class(primal_optimizer, **scheduler_kwargs)

    for step_id in range(7):
        coop.zero_grad()
        lagrangian = formulation.compute_lagrangian(cmp.closure, params)
        formulation.backward(lagrangian)

        if hasattr(primal_optimizer, "extrapolation"):
            # Only one dual_scheduler step should be performed even if
            # extrapolation is used
            coop.step(cmp.closure, params)
        else:
            coop.step()

        primal_scheduler.step()
        coop.dual_scheduler.step()

        # Check that the dual learning rate is decreasing correctly
        primal_lr = primal_scheduler.get_last_lr()
        dual_lr = coop.dual_scheduler.get_last_lr()

        if scheduler_name == "ExponentialLR":
            target_lr = torch.tensor(0.1) ** step_id
        elif scheduler_name == "ConstantLR":
            # Need to add 1 to step_id since we are performing the check at the
            # _end_ of the epoch
            target_lr = torch.tensor(10.0) if (1 + step_id) >= 4 else torch.tensor(5.0)

        assert torch.allclose(torch.tensor(primal_lr), target_lr)
        assert torch.allclose(torch.tensor(dual_lr), target_lr)
