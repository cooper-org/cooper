import torch

import cooper


class CustomCMP(cooper.ConstrainedMinimizationProblem):
    def __init__(self, is_constrained=False):
        super().__init__(is_constrained)

    def update_state(self, params, use_ineq=False, use_proxy_ineq=False):
        """Define toy `cmp` function"""

        param_x, param_y = params
        self.loss = param_x ** 2 + 2 * param_y ** 2

        if use_ineq:
            # Two inequality constraints
            self.ineq_defect = torch.stack(
                [
                    -param_x - param_y + 1.0,  # x + y \ge 1
                    param_x ** 2 + param_y - 1.0,  # x**2 + y \le 1.0
                ]
            )

            if use_proxy_ineq:
                # Using **slightly** different functions for the proxy constraints
                self.proxy_ineq_defect = torch.stack(
                    [
                        -0.9 * param_x - param_y + 1.0,  # x + y \ge 1
                        param_x ** 2 + 0.9 * param_y - 1.0,  # x**2 + y \le 1.0
                    ]
                )
            else:
                self.proxy_ineq_defect = None

        else:
            self.ineq_defect = None
            self.proxy_ineq_defect = None
