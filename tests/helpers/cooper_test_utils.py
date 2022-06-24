"""Cooper-related utilities for writing tests."""

import functools
from dataclasses import dataclass
from typing import Union

import pytest
import testing_utils
import torch

import cooper


@dataclass
class TestProblemData:
    params: Union[torch.Tensor, torch.nn.Module]
    cmp: cooper.ConstrainedMinimizationProblem
    coop: cooper.ConstrainedOptimizer
    formulation: cooper.problem.Formulation
    device: torch.device
    mktensor: callable

    def as_tuple(self):
        field_names = ["params", "cmp", "coop", "formulation", "device", "mktensor"]
        return (getattr(self, _) for _ in field_names)


def build_test_problem(
    aim_device,
    primal_optim_cls,
    primal_init,
    dual_optim_cls,
    use_ineq,
    use_proxy_ineq,
    dual_restarts,
    alternating,
    primal_optim_kwargs={"lr": 1e-2},
    dual_optim_kwargs={"lr": 1e-2},
    dual_scheduler=None,
    primal_model=None,
):

    # Retrieve available device, and signal to skip test if GPU is not available
    device, skip = testing_utils.get_device_skip(aim_device, torch.cuda.is_available())

    if skip.do_skip:
        pytest.skip(skip.skip_reason)

    cmp = Toy2dCMP(use_ineq=use_ineq, use_proxy_ineq=use_proxy_ineq)

    if primal_init is None:
        primal_model.to(device)
        primal_optimizer = primal_optim_cls(
            primal_model.parameters(), **primal_optim_kwargs
        )
        params = primal_model.parameters()
    else:
        params = torch.nn.Parameter(torch.tensor(primal_init, device=device))
        primal_optimizer = primal_optim_cls([params], **primal_optim_kwargs)

    if use_ineq:
        # Constrained case
        dual_optimizer = cooper.optim.partial_optimizer(
            dual_optim_cls, **dual_optim_kwargs
        )
        formulation = cooper.LagrangianFormulation(cmp)
    else:
        # Unconstrained case
        dual_optimizer = None
        formulation = cooper.UnconstrainedFormulation(cmp)

    coop = cooper.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optimizer,
        dual_optimizer=dual_optimizer,
        dual_scheduler=dual_scheduler,
        dual_restarts=dual_restarts,
        alternating=alternating,
    )

    # Helper function to instantiate tensors in correct device
    mktensor = functools.partial(torch.tensor, device=device)

    return TestProblemData(params, cmp, coop, formulation, device, mktensor)


class Toy2dCMP(cooper.ConstrainedMinimizationProblem):
    """
    Simple test on a 2D quadratically-constrained quadratic programming problem
        min x**2 + 2*y**2
        st.
            x + y >= 1
            x**2 + y <= 1

    If proxy constrainst are used, the "differentiable" surrogates are:
            0.9 * x + y >= 1
            x**2 + 0.9 * y <= 1

    This is a convex optimization problem.

    The constraint levels of the differentiable surrogates are not strictly
    required since these functions are only employed via their gradients, thus
    the constant contribution of the constraint level disappears. We include
    them here for readability.

    Verified solution from WolframAlpha of the original constrained problem:
        (x=2/3, y=1/3)
    Link to WolframAlpha query: https://tinyurl.com/ye8dw6t3
    """

    def __init__(self, use_ineq=False, use_proxy_ineq=False):
        self.use_ineq = use_ineq
        self.use_proxy_ineq = use_proxy_ineq
        super().__init__(is_constrained=self.use_ineq)

    def closure(self, params):

        if isinstance(params, torch.nn.Module):
            param_x, param_y = params.forward()
        else:
            param_x, param_y = params

        loss = param_x**2 + 2 * param_y**2

        # No equality constraints
        eq_defect = None

        if self.use_ineq:
            # Two inequality constraints
            ineq_defect = torch.stack(
                [
                    -param_x - param_y + 1.0,  # x + y \ge 1
                    param_x**2 + param_y - 1.0,  # x**2 + y \le 1.0
                ]
            )

            if self.use_proxy_ineq:
                # Using **slightly** different functions for the proxy
                # constraints
                proxy_ineq_defect = torch.stack(
                    [
                        # Orig constraint: x + y \ge 1
                        -0.9 * param_x - param_y + 1.0,
                        # Orig constraint: x**2 + y \le 1.0
                        param_x**2 + 0.9 * param_y - 1.0,
                    ]
                )
            else:
                proxy_ineq_defect = None

        else:
            ineq_defect = None
            proxy_ineq_defect = None

        closure_state = cooper.CMPState(
            loss=loss,
            eq_defect=eq_defect,
            ineq_defect=ineq_defect,
            proxy_ineq_defect=proxy_ineq_defect,
        )

        return closure_state


def get_optimizer_from_str(optimizer_str):
    """
    Returns an optimizer class from the string name of the optimizer.
    """
    try:
        return getattr(cooper.optim, optimizer_str)
    except:
        return getattr(torch.optim, optimizer_str)
