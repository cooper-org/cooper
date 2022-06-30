import pdb

import torch

import cooper


class Toy2dCMP(cooper.ConstrainedMinimizationProblem):
    def __init__(self, use_ineq=False, use_proxy_ineq=False):
        self.use_ineq = use_ineq
        self.use_proxy_ineq = use_proxy_ineq
        super().__init__(is_constrained=self.use_ineq)

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
                        -0.9 * param_x - param_y + 1.0,  # x + y \ge 1
                        param_x**2 + 0.9 * param_y - 1.0,  # x**2 + y \le 1.0
                    ]
                )
            else:
                proxy_ineq_defect = None

        else:
            ineq_defect = None
            proxy_ineq_defect = None

        return cooper.CMPState(
            loss=None,
            eq_defect=eq_defect,
            ineq_defect=ineq_defect,
            proxy_ineq_defect=proxy_ineq_defect,
        )
