r"""
Linear classification with rate constraints
===============================================

In this example we consider a linear classification problem on synthetically
mixture of Gaussians data. We constraint the model to predict at least 70% of
the training points as blue.

Note that this is a non-differentiable constraint, and thus the typical
Lagrangian approach is not applicable as it requires to compute the derivatives
of the constraints :math:`g` and :math:`h`.

A commonly used approach to deal with this difficulty is to retain the
Lagrangian formulation, but replace the constraints with differentiable
approximations or surrogates. However, changing the constraint functions can
result in an over- or under-constrained version of the problem.

:cite:t:`cotter2019JMLR` propose a *proxy-Lagrangian formulation*, in which the
non-differentiable constraints are relaxed *only when necessary*. In other
words, the non differentiable constraint functions are used to compute the
Lagrangian and constraint violations (and thus the multiplier updates), while
the surrogates are used to compute the primal gradients.

This example is based on Fig. 2 of :cite:t:`cotter2019JMLR`. Here we present
the naive setting where proxy-constraints are used on a Lagrangian formulation,
rather than a "proper" proxy-Lagrangian formulation. For details on the
notion of a proxy-Lagrangian formulation, see :math:`\S` 4.2 in
:cite:p:`cotter2019JMLR`.
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import style_utils
import torch

import cooper

style_utils.set_plot_style()

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def generate_mog_dataset():
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


def plot_pane(ax, inputs, x1, x2, achieved_const, titles, colors):
    const_str = str(np.round(achieved_const, 0)) + "%"
    ax.scatter(*torch.transpose(inputs, 0, 1), color=colors)
    ax.plot(x1, x2, color="gray", linestyle="--")
    ax.fill_between(x1, -2, x2, color=blue, alpha=0.1)
    ax.fill_between(x1, x2, 2, color=red, alpha=0.1)

    ax.set_aspect("equal")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_title(titles[idx] + " - Pred. Blue %: " + const_str)


class UnconstrainedMixtureSeparation(cooper.ConstrainedMinimizationProblem):
    def __init__(self):
        super().__init__()

    def compute_cmp_state(self, model, inputs, targets):
        logits = model(inputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.flatten(), targets)
        return cooper.CMPState(loss=loss)


class MixtureSeparation(cooper.ConstrainedMinimizationProblem):
    """
    Implements CMP for separating the MoG dataset with a linear predictor.

    Args:
        use_proxy: Flag to use proxy-constraints. If ``True``, we use a hinge
            relaxation. Defaults to ``False``.
        constraint_level: Minimum proportion of points to be predicted as belonging to
            the blue class. Ignored when ``is_constrained==False``. Defaults to ``0.7``.
    """

    def __init__(self, use_strict_constraints: bool = False, constraint_level: float = 0.7):
        super().__init__()

        constraint_type = constraint_type = cooper.ConstraintType.INEQUALITY
        self.multiplier = cooper.multipliers.DenseMultiplier(constraint_type=constraint_type, num_constraints=1)
        self.constraint = cooper.Constraint(
            constraint_type=constraint_type,
            formulation_type=cooper.FormulationType.LAGRANGIAN,
            multiplier=self.multiplier,
        )

        self.constraint_level = constraint_level
        self.use_strict_constraints = use_strict_constraints

    def compute_cmp_state(self, model, inputs, targets):
        logits = model(inputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.flatten(), targets)

        # Separating classes s.t. predicting at least constraint_level as class 0

        # Hinge approximation of the rate
        probs = torch.sigmoid(logits)

        # The differentiable violation uses the hinge loss as a surrogate
        hinge = torch.mean(torch.max(torch.zeros_like(probs), 1 - probs))
        differentiable_violation = -hinge

        strict_violation = None
        if self.use_strict_constraints:
            # Use strict violations to update the Lagrange multipliers
            classes = logits >= 0.0
            prop_0 = torch.sum(classes == 0) / targets.numel()
            strict_violation = self.constraint_level - prop_0

        constraint_state = cooper.ConstraintState(violation=differentiable_violation, strict_violation=strict_violation)
        return cooper.CMPState(loss=loss, observed_constraints=[(self.constraint, constraint_state)])


def train(problem_name, inputs, targets, num_iters=5000, lr=1e-2, constraint_level=0.7):
    """
    Train via SGD
    """

    is_constrained = "const" in problem_name.lower()
    use_strict_constraints = "proxy" in problem_name.lower()

    model = torch.nn.Linear(2, 1)
    primal_optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.7)

    if is_constrained:
        cmp = MixtureSeparation(use_strict_constraints, constraint_level)
        dual_optimizer = torch.optim.SGD(cmp.multiplier.parameters(), lr=lr, momentum=0.7, maximize=True)
        cooper_optimizer = cooper.optim.SimultaneousOptimizer(
            primal_optimizers=primal_optimizer, dual_optimizers=dual_optimizer, multipliers=cmp.multiplier
        )
    else:
        cmp = UnconstrainedMixtureSeparation()
        cooper_optimizer = cooper.optim.UnconstrainedOptimizer(primal_optimizers=primal_optimizer, cmp=cmp)

    for _ in range(num_iters):
        cmp_state, lagrangian_store = cooper_optimizer.roll(
            compute_cmp_state_kwargs=dict(model=model, inputs=inputs, targets=targets)
        )

    # Number of elements predicted as class 0 in the train set after training
    logits = model(inputs)
    pred_classes = logits >= 0.0
    prop_0 = torch.sum(pred_classes == 0) / targets.numel()

    return model, 100 * prop_0.item()


# Plot configs
titles = ["Unconstrained", "Const+Surrogate", "Const+Proxy"]
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Data and training configs
inputs, labels = generate_mog_dataset()
constraint_level = 0.7
lr = 2e-2
num_iters = 5000

for idx, name in enumerate(titles):
    model, achieved_const = train(name, inputs, labels, lr=lr, num_iters=num_iters, constraint_level=constraint_level)

    # Compute decision boundary
    weight, bias = model.weight.data.flatten().numpy(), model.bias.data.numpy()
    x1 = np.linspace(-2, 2, 100)
    x2 = (-1 / weight[1]) * (weight[0] * x1 + bias)

    # Color points according to true label
    red, blue = style_utils.COLOR_DICT["red"], style_utils.COLOR_DICT["blue"]
    colors = [red if _ == 1 else blue for _ in labels.flatten()]
    plot_pane(axs[idx], inputs, x1, x2, achieved_const, titles, colors)

fig.suptitle("Goal: Predict at least " + str(constraint_level * 100) + "% as blue")
plt.show()
