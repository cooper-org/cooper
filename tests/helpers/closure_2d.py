import torch
import torch_coop


def construct_closure(params, use_ineq=False, use_proxy_ineq=False):
    param_x, param_y = params

    def closure_fn():
        # Define toy closure function

        loss = param_x ** 2 + 2 * param_y ** 2

        # No equality constraints
        eq_defect = None

        if use_ineq:
            # Two inequality constraints
            ineq_defect = torch.stack(
                [
                    -param_x - param_y + 1.0,  # x + y \ge 1
                    param_x ** 2 + param_y - 1.0,  # x**2 + y \le 1.0
                ]
            )

            if use_proxy_ineq:
                # Using **slightly** different functions for the proxy constraints
                proxy_ineq_defect = torch.stack(
                    [
                        -0.9 * param_x - param_y + 1.0,  # x + y \ge 1
                        param_x ** 2 + 0.9 * param_y - 1.0,  # x**2 + y \le 1.0
                    ]
                )
            else:
                proxy_ineq_defect = None

        else:
            ineq_defect = None
            proxy_ineq_defect = None

        closure_state = torch_coop.CMPState(
            loss=loss,
            eq_defect=eq_defect,
            ineq_defect=ineq_defect,
            proxy_ineq_defect=proxy_ineq_defect,
        )

        return closure_state

    return closure_fn
