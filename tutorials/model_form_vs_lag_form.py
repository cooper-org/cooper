"""This module is solely intended to compare the performance of the
LagrangianModelFormulation against the regular LagrangianFormulation. The results
will be logged to a wandb team project. Please note that wandb is not a dependecy of the
project so you will have to install it manually."""

from dataclasses import dataclass
import functools
import os
import random
from types import GeneratorType
from typing import Union

import cooper
import numpy as np
import torch
import wandb

random.seed(121212)
np.random.seed(121212)
torch.manual_seed(121212)
torch.cuda.manual_seed(121211)

config = {
    "use_mult_model": True,
    "primal_lr": 1e-2,
    "dual_lr": 1e-2,
    "use_wandb": True,
    "use_wandb_offline": False,
}


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
    device,
    primal_optim_cls,
    primal_init,
    dual_optim_cls,
    use_ineq,
    use_proxy_ineq,
    use_mult_model,
    dual_restarts,
    alternating,
    primal_optim_kwargs={"lr": 1e-2},
    dual_optim_kwargs={"lr": 1e-2},
    dual_scheduler=None,
    primal_model=None,
    formulation_cls=cooper.LagrangianFormulation,
):

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

    cmp = Toy2dCMP(
        device,
        use_ineq=use_ineq,
        use_proxy_ineq=use_proxy_ineq,
        use_mult_model=use_mult_model,
    )

    if use_mult_model or use_ineq:
        # Constrained case
        dual_optimizer = cooper.optim.partial_optimizer(
            dual_optim_cls, **dual_optim_kwargs
        )

    if use_mult_model:
        # Exclusive for the model formulation
        ineq_multiplier_model = ToyMultiplierModel(3, 10, device)
        formulation = cooper.formulation.LagrangianModelFormulation(
            cmp, ineq_multiplier_model=ineq_multiplier_model
        )

    elif use_ineq:
        # Constrained case different from model formulation
        formulation = formulation_cls(cmp)

    else:
        # Unconstrained case
        dual_optimizer = None
        formulation = cooper.UnconstrainedFormulation(cmp)

    cooper_optimizer_kwargs = {
        "formulation": formulation,
        "primal_optimizers": primal_optimizers,
        "dual_optimizer": dual_optimizer,
        "dual_scheduler": dual_scheduler,
        "extrapolation": "Extra" in str(primal_optimizers[0]),
        "alternating": alternating,
        "dual_restarts": dual_restarts,
    }

    coop = cooper.optim.create_optimizer_from_kwargs(**cooper_optimizer_kwargs)

    # Helper function to instantiate tensors in correct device
    mktensor = functools.partial(torch.tensor, device=device)

    return TestProblemData(params, cmp, coop, formulation, device, mktensor)


class ToyMultiplierModel(cooper.multipliers.MultiplierModel):
    """
    Simplest MultiplierModel possible, a linear model with a single output.
    """

    def __init__(self, n_params, n_hidden_units, device):
        super().__init__()
        self.linear1 = torch.nn.Linear(n_params, n_hidden_units, device=device)
        self.linear2 = torch.nn.Linear(n_hidden_units, 1, device=device)

    def forward(self, constraint_features: torch.Tensor):
        x = self.linear1(constraint_features)
        x = torch.relu(x)
        x = self.linear2(x)
        return torch.reshape(torch.nn.functional.relu(x), (-1,))


def get_optimizer_from_str(optimizer_str):
    """
    Returns an optimizer class from the string name of the optimizer.
    """
    try:
        return getattr(cooper.optim, optimizer_str)
    except:
        return getattr(torch.optim, optimizer_str)


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

    This problem is designed to be used with the Lagrangian Model, thus, we define
    constraint features to feed into the `Multiplier Model`. The first two features
    correspont to the exponent of the `x` and `y` variables, respectively. The last
    feature correspond to the slack term and the direction of the inequality constraint
    (i.e. `-1` for `>=` and `1` for `<=`).

    Verified solution from WolframAlpha of the original constrained problem:
        (x=2/3, y=1/3)
    Link to WolframAlpha query: https://tinyurl.com/ye8dw6t3
    """

    def __init__(
        self, device, use_ineq=False, use_proxy_ineq=False, use_mult_model=False
    ):
        self.use_ineq = use_ineq
        self.use_proxy_ineq = use_proxy_ineq
        self.use_mult_model = use_mult_model
        self.device = device

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

    def closure(self, params, use_third_constraint):

        cmp_state = self.defect_fn(params, use_third_constraint)
        cmp_state.loss = self.loss_fn(params)

        return cmp_state

    def loss_fn(self, params):
        param_x, param_y = self.eval_params(params)

        return param_x**2 + 2 * param_y**2

    def defect_fn(self, params, use_third_constraint):

        param_x, param_y = self.eval_params(params)

        # No equality constraints
        eq_defect = None

        if self.use_ineq or self.use_mult_model:
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

        if use_third_constraint:
            # Append a third constraint
            third_defect = torch.tensor([param_x + param_y - 1.0], device=self.device)
            ineq_defect = torch.cat([ineq_defect, third_defect])

            if self.use_proxy_ineq:
                proxy_third_defect = torch.tensor(
                    [-0.9 * param_x - param_y + 1.0], device=self.device
                )
                proxy_ineq_defect = torch.cat([proxy_ineq_defect, proxy_third_defect])

        # Create inequality constraint features. The first feature is the exponent for
        # the x, the second for the y, and the third is the slack term. The sign of the
        # slack term depends on the constraint type (i.e. >= or <=).
        misc = None
        if self.use_mult_model:
            misc = {"ineq_constraint_features": self.constraint_features}

        return cooper.CMPState(
            loss=None,
            eq_defect=eq_defect,
            ineq_defect=ineq_defect,
            proxy_ineq_defect=proxy_ineq_defect,
            misc=misc,
        )


def main(device, use_mult_model, primal_lr, dual_lr):

    if config["use_wandb"]:
        wandb.init(entity="many-constraints", project="toy-problem", config=config)

    if use_mult_model:
        formulation_cls = cooper.formulation.LagrangianModelFormulation
    else:
        formulation_cls = cooper.formulation.LagrangianFormulation

    test_problem_data = build_test_problem(
        device=device,
        primal_optim_cls=torch.optim.SGD,
        primal_init=[0.0, -1.0],
        dual_optim_cls=torch.optim.SGD,
        use_ineq=True,
        use_proxy_ineq=False,
        use_mult_model=use_mult_model,
        dual_restarts=False,
        alternating=False,
        primal_optim_kwargs={"lr": primal_lr},
        dual_optim_kwargs={"lr": dual_lr},
        dual_scheduler=None,
        formulation_cls=formulation_cls,
    )

    params, cmp, coop, formulation, _, _ = test_problem_data.as_tuple()

    if use_mult_model:
        coop.instantiate_dual_optimizer_and_scheduler()

    use_third_constraint = False
    for step_id in range(500):
        coop.zero_grad()
        lagrangian = formulation.compute_lagrangian(
            closure=cmp.closure,
            params=params,
            use_third_constraint=use_third_constraint,
        )
        formulation.backward(lagrangian)
        coop.step(cmp.closure, params, use_third_constraint)

        if config["use_wandb"]:
            logs = metric_logger(cmp.state, formulation, lagrangian, params)
            wandb.log(logs, step_id)

        if step_id == 400:
            use_third_constraint = True
            if use_mult_model:
                cmp.constraint_features = torch.cat(
                    (
                        cmp.constraint_features,
                        torch.tensor([[1.0, 1.0, 1.0]], device=device),
                    )
                )
            else:
                # Get the current multipliers
                ineq_mult = formulation.state()[0]
                # Third multiplier init
                third_mult = torch.tensor([0.0], device=device)
                # Initialize a new multiplier tensor and set the third constraint
                # multiplier to zero
                new_ineq_mult_init = torch.cat((ineq_mult, third_mult), dim=-1)
                # Reinitialize the formulation with new multipliers
                formulation = cooper.LagrangianFormulation(cmp, new_ineq_mult_init)

        if step_id % 10 == 0:
            print(f"Iteration {step_id}")


def metric_logger(cmp_state, formulation, lagrangian, params):

    if isinstance(formulation, cooper.formulation.LagrangianModelFormulation):
        multipliers = formulation.state(
            ineq_features=cmp_state.misc["ineq_constraint_features"]
        )[0]
    else:
        multipliers = formulation.state()[0]

    logs = {
        "loss": cmp_state.loss.item(),
        "lagrangian": lagrangian.item(),
        "ineq_defect": wandb.Histogram(_to_cpu_numpy(cmp_state.ineq_defect)),
        "ineq_multipliers": wandb.Histogram(_to_cpu_numpy(multipliers)),
        "x": params[0].item(),
        "y": params[1].item(),
    }

    return logs


def _to_cpu_numpy(
    x: Union[torch.Tensor, list, dict[str, torch.Tensor]]
) -> Union[np.ndarray, list, dict[str, np.ndarray]]:

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, list):
        return [_to_cpu_numpy(v) for v in x]
    elif isinstance(x, dict):
        return {k: _to_cpu_numpy(v) for k, v in x.items()}
    else:
        raise ValueError(f"Unknown type: {type(x)}")


def setup_wandb_logging(config):

    # Configure WandB settings
    if not config["use_wandb"]:
        wandb_settings = wandb.Settings(
            mode="disabled",
            program=__name__,
            program_relpath=__name__,
            disable_code=True,
        )
        wandb.setup(wandb_settings)
    else:
        if config["use_wandb_offline"]:
            os.environ["WANDB_MODE"] = "offline"


if __name__ == "__main__":
    if torch.cuda.is_available():
        main("cuda", config["use_mult_model"], config["primal_lr"], config["dual_lr"])
    else:
        main("cpu", config["use_mult_model"], config["primal_lr"], config["dual_lr"])
