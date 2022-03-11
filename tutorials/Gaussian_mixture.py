"""
Linear classifier on a mixture of Gaussians with rate constraints.
Based on Fig. 2 of A. Cotter, H. Jiang, M. Gupta, S. Wang, T. Narayan, S. You,
and K. Sridharan. Optimization with Non-Differentiable
Constraints with Applications to Fairness, Recall, Churn,
and Other Goals. In JMLR, 2019.
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import style_utils
import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss

import cooper
from cooper.optim import SGD

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def gen_dataset():
    """
    Generate a MoG dataset on 2D, with two classes.
    """

    n_per_class = 100
    dim = 2
    n_gaussians = 4
    mus = [(0, 1), (-1, 0), (0, -1), (1, 0)]
    mus = [torch.tensor(m) for m in mus]
    var = 0.05

    inputs, labels = [], []

    for id in range(n_gaussians):
        # Generate input data by mu + x @ sqrt(cov)
        cov = np.sqrt(var) * torch.eye(dim)
        mu = mus[id]
        inputs.append(mu + torch.randn(n_per_class, dim) @ cov)

        # Labels
        labels.append(torch.tensor(n_per_class * [1.0 if id < 2 else 0.0]))

    return torch.cat(inputs, dim=0), torch.cat(labels, dim=0)


class MixtureSeparation(cooper.ConstrainedMinimizationProblem):
    """
    Implements CMP for separating the MoG dataset with a linear predictor.
    The CMP formulation can be unconstrained: minimizing the prediction loss,
    constrained: minimizing the loss s.t. predicting at least const_level
    proportion of the data as class 0. In the constrained case, proxy=False
    means that the constraint is based on a hinge relaxation of the rate of
    predictions as class 0; proxy=True means that the non-proxy rate is used as
    to update the Lagrange multipliers.
    """

    def __init__(self, is_constrained, use_proxy=False, const_level=0.7):

        super().__init__(is_constrained=is_constrained)

        self.const_level = const_level
        self.use_proxy = use_proxy

        # Linear predictor
        self.linear = torch.nn.Linear(2, 1)

    def closure(self, inputs, targets):

        logits = self.linear(inputs)
        loss = bce_loss(logits.flatten(), targets)

        if not self.is_constrained:
            # Unconstrained problem of separating two classes
            state = cooper.CMPState(
                loss=loss,
            )

        if self.is_constrained:
            # Separating classes s.t. predicting at least const_level as class 0

            # Hinge approximation of the rate
            probs = torch.sigmoid(logits)
            hinge = torch.mean(torch.max(torch.zeros_like(probs), 1 - probs))
            proxy_defect = self.const_level - hinge  # level - proxy_defect <= 0

            if not self.use_proxy:
                # Use a proxy for the constraint: a hinge relaxation
                state = cooper.CMPState(
                    loss=loss,
                    ineq_defect=proxy_defect,
                )
            else:
                # Use non-proxy defects to update the Lagrange multipliers

                # Proportion of elements in class 0 is the non-proxy defect
                classes = logits >= 0.0
                prop_0 = torch.sum(classes == 0) / targets.numel()
                actual_defect = self.const_level - prop_0

                state = cooper.CMPState(
                    loss=loss,
                    ineq_defect=actual_defect,
                    proxy_ineq_defect=proxy_defect,
                )

        return state


def train(problem, inputs, targets, n_iters=10000, lr=1e-2, const_level=0.7):
    """
    Train via SGD
    """

    is_constrained = problem == "constrained" or problem == "proxy"
    use_proxy = problem == "proxy"

    cmp = MixtureSeparation(is_constrained, use_proxy, const_level)
    formulation = cooper.LagrangianFormulation(cmp)

    primal_optimizer = SGD(cmp.linear.parameters(), lr=lr)
    dual_optimizer = cooper.optim.partial(SGD, lr=lr) if is_constrained else None
    optimizer = cooper.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optimizer,
        dual_optimizer=dual_optimizer,
    )

    for i in range(n_iters):
        optimizer.zero_grad()
        if is_constrained:
            lagrangian = formulation.composite_objective(cmp.closure, inputs, targets)
            formulation.custom_backward(lagrangian)
        else:
            # No Lagrangian in the unconstrained case
            loss = cmp.closure(inputs, targets).loss
            loss.backward()

        optimizer.step()

    # Number of elements predicted as class 0 in the train set after training
    logits = cmp.linear(inputs)
    classes = logits >= 0.0
    prop_0 = torch.sum(classes == 0) / targets.numel()

    return cmp.linear, prop_0.item()


def main():
    """
    Plot the resulting boundary of the linear predictor across three different
    approaches: unconstrained, constrained, and constrained with proxy consts.
    """

    const_level = 0.7
    inputs, labels = gen_dataset()
    titles = ["Unconstrained", "Constrained", "Proxy"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    red, blue = style_utils.COLOR_DICT["red"], style_utils.COLOR_DICT["blue"]
    colors = [red if _ == 1 else blue for _ in labels.flatten()]

    for idx, name in enumerate(titles):

        model, achieved_const = train(name, inputs, labels, const_level=const_level)

        weight = model.weight.data.flatten().numpy()
        bias = model.bias.data.numpy()

        # Reparametrize boundary as x2 = m * x1 + b
        m = -weight[0] / weight[1]
        b = -bias / weight[1]

        # Plot data and boundary
        x1 = np.linspace(-2, 2, 100)
        x2 = m * x1 + b

        ax = axs[idx]

        ax.scatter(*torch.transpose(inputs, 0, 1), color=colors)
        ax.plot(x1, x2, color="gray", linestyle="--")
        ax.fill_between(x1, -2, x2, color=blue, alpha=0.1)
        ax.fill_between(x1, x2, 2, color=red, alpha=0.1)

        ax.set_aspect("equal")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_title(
            titles[idx] + " (Blue preds: " + str(np.round(achieved_const, 2)) + "%)"
        )
        fig.suptitle("Predict at least " + str(const_level * 100) + "% as blue")

    fig.show()


if __name__ == "__main__":
    main()
