"""Utils for Lagrangian Model testing."""


from dataclasses import dataclass
import functools
from types import GeneratorType
from typing import Union

import cooper
import pytest
import torch

import testing_utils


@dataclass
class TestProblemData:
    params: Union[torch.Tensor, torch.nn.Module]
    cmp: cooper.ConstrainedMinimizationProblem
    coop: cooper.ConstrainedOptimizer
    formulation: cooper.Formulation
    device: torch.device
    mktensor: callable

    def as_tuple(self):
        field_names = ["params", "cmp", "coop", "formulation", "device", "mktensor"]
        return (getattr(self, _) for _ in field_names)


def build_test_problem(
    aim_device: bool,
    sample_constraints: bool,
    primal_init,
    primal_optim_cls,
    dual_optim_cls,
    primal_optim_kwargs={"lr": 1e-2},
    dual_optim_kwargs={"lr": 1e-2},
    primal_model=None,
    formulation_cls=cooper.formulation.LagrangianModelFormulation,
    dual_scheduler=None,
):
    """Build a test problem for the Lagrangian Model."""

    # Retrieve available device, and signal to skip test if GPU is not available
    device, skip = testing_utils.get_device_skip(aim_device, torch.cuda.is_available())

    if skip.do_skip:
        pytest.skip(skip.skip_reason)

    cmp = Toy2dLM(sample_constraints=sample_constraints, device=device)

    if primal_init is None:
        primal_model.to(device)
        params = primal_model.parameters()
        params_ = params
    else:
        params = torch.nn.Parameter(torch.tensor(primal_init, device=device))
        params_ = [params]

    if isinstance(primal_optim_cls, list):
        # params is created in a different way to avoid slicing issues with the
        # autograd engine. Data contents of params are not modified.
        sliceable_params = (
            list(params)[0] if isinstance(params, GeneratorType) else params
        )
        params = [torch.nn.Parameter(_) for _ in sliceable_params.data]
        params_ = params

        primal_optimizers = []
        for p, cls, kwargs in zip(params, primal_optim_cls, primal_optim_kwargs):
            primal_optimizers.append(cls([p], **kwargs))

    else:
        primal_optimizers = [primal_optim_cls(params_, **primal_optim_kwargs)]

    dual_optimizer = cooper.optim.partial_optimizer(dual_optim_cls, **dual_optim_kwargs)

    ineq_multiplier_model = ToyMultiplierModel(3, 10)
    if device == "cuda":
        ineq_multiplier_model.cuda()

    formulation = formulation_cls(cmp, ineq_multiplier_model=ineq_multiplier_model)

    cooper_optimizer_kwargs = {
        "formulation": formulation,
        "primal_optimizers": primal_optimizers,
        "dual_optimizer": dual_optimizer,
        "dual_scheduler": dual_scheduler,
        "extrapolation": False,  # use simultaneous optimizer for the tests
        "alternating": False,
        "dual_restarts": False,  # multiplier model cannnot be reset
    }

    coop = cooper.optim.create_optimizer_from_kwargs(**cooper_optimizer_kwargs)

    # Helper function to instantiate tensors in correct device
    mktensor = functools.partial(torch.tensor, device=device)

    return TestProblemData(params, cmp, coop, formulation, device, mktensor)


class ToyMultiplierModel(cooper.multipliers.MultiplierModel):
    """
    Simplest MultiplierModel possible, a linear model with a single output.
    """

    def __init__(self, n_params, n_hidden_units):
        super().__init__()
        self.linear1 = torch.nn.Linear(n_params, n_hidden_units)
        self.linear2 = torch.nn.Linear(n_hidden_units, 1)


    def forward(self, constraint_features: torch.Tensor):
        x  = self.linear1(constraint_features)
        x = torch.relu(x)
        x = self.linear2(x)
        return torch.nn.functional.relu(x)


class Toy2dLM(cooper.ConstrainedMinimizationProblem):
    """
    Simple test on a 2D quadratically-constrained quadratic programming problem
        min x**2 + 2*y**2
        st.
            x + y >= 1
            x**2 + y <= 1

    This is a convex optimization problem.

    This problem is designed to be used with the Lagrangian Model, thus, we define
    constraint features to feed into the `Multiplier Model`. The first two features
    correspont to the exponent of the `x` and `y` variables, respectively. The last
    feature correspond to the slack term and the direction of the inequality constraint
    (i.e. `-1` for `>=` and `1` for `<=`).

    Verified solution from WolframAlpha of the original constrained problem:
        (x=2/3, y=1/3)
    Link to WolframAlpha query: https://tinyurl.com/ye8dw6t3
    """

    def __init__(self, sample_constraints: bool, device: torch.device):
        self.sample_constraints = sample_constraints

        # Define constraint features
        self.constraint_features = torch.tensor(
            [[1.0, 1.0, -1.0], [2.0, 1.0, 1.0]], device=device
        )
        super().__init__()

    def eval_params(self, params):
        if isinstance(params, torch.nn.Module):
            param_x, param_y = params.forward()
        else:
            param_x, param_y = params

        return param_x, param_y

    def closure(self, params):

        cmp_state = self.defect_fn(params)
        cmp_state.loss = self.loss_fn(params)

        return cmp_state

    def loss_fn(self, params):
        param_x, param_y = self.eval_params(params)

        return param_x**2 + 2 * param_y**2

    def defect_fn(self, params):

        param_x, param_y = self.eval_params(params)

        # Loss
        loss = param_x**2 + 2 * param_y**2

        # Two inequality constraints
        ineq_defect = torch.stack(
            [
                -param_x - param_y + 1.0,  # x + y \ge 1
                param_x**2 + param_y - 1.0,  # x**2 + y \le 1.0
            ]
        )

        if self.sample_constraints:
            # Random number for sampling
            rand = round(torch.rand(1).item())

            sampled_ineq_defect = ineq_defect[rand]
            sampled_constraint_features = self.constraint_features[rand]

        else:
            sampled_ineq_defect = ineq_defect
            sampled_constraint_features = self.constraint_features

        return cooper.CMPState(
            loss=loss,
            ineq_defect=sampled_ineq_defect,
            misc={"ineq_constraint_features": sampled_constraint_features},
        )
