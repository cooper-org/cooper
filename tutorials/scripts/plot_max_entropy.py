"""
Finding a discrete maximum entropy distribution
===============================================

Here we consider a simple convex optimization problem to illustrate how to use
**Cooper**. This example is inspired by `this StackExchange question
<https://datascience.stackexchange.com/questions/107366/how-do-you-solve-strictly-constrained-optimization-problems-with-pytorch>`_\:

*I am trying to solve the following problem using Pytorch: given a 6-sided die
whose average roll is known to be 4.5, what is the maximum entropy distribution
for the faces?*
"""

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from style_utils import *

import cooper
from cooper import CMPState, ConstraintGroup, ConstraintState, ConstraintType, FormulationType

torch.manual_seed(0)
np.random.seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MaximumEntropy(cooper.ConstrainedMinimizationProblem):
    def __init__(self, target_mean: float) -> None:
        self.target_mean = target_mean

        multiplier_kwargs = {"shape": 1, "device": DEVICE}
        constraint_kwargs = {"constraint_type": ConstraintType.EQUALITY, "formulation_type": FormulationType.LAGRANGIAN}
        self.mean_constraint = ConstraintGroup(**constraint_kwargs, multiplier_kwargs=multiplier_kwargs)
        self.sum_constraint = ConstraintGroup(**constraint_kwargs, multiplier_kwargs=multiplier_kwargs)

        self.all_constraints = [self.sum_constraint, self.mean_constraint]

        super().__init__()

    def compute_cmp_state(self, log_probs: torch.Tensor) -> CMPState:
        probs = torch.exp(log_probs)
        # Loose negative sign since we want to *maximize* the entropy
        entropy = torch.sum(probs * log_probs)

        # Equality constraints for proper normalization and mean constraint
        mean = torch.sum(probs * torch.arange(1, len(probs) + 1, device=DEVICE))

        self.sum_constraint.state = ConstraintState(violation=torch.sum(probs) - 1)
        self.mean_constraint.state = ConstraintState(violation=mean - self.target_mean)

        return CMPState(loss=entropy, observed_constraints=self.all_constraints)


# Define the problem with the constraint groups
cmp = MaximumEntropy(target_mean=4.5)

# Define the primal parameters and optimizer
log_probs = torch.nn.Parameter(torch.log(torch.ones(6, device=DEVICE) / 6))
primal_optimizer = cooper.optim.ExtraSGD([log_probs], lr=3e-2)

# Define the dual optimizer
dual_parameters = []
[dual_parameters.extend(_.multiplier.parameters()) for _ in cmp.all_constraints]
dual_optimizer = cooper.optim.ExtraSGD(dual_parameters, lr=5e-2, maximize=True)

# Wrap the formulation and both optimizers inside a ExtrapolationConstrainedOptimizer
cooper_optimizer = cooper.optim.ExtrapolationConstrainedOptimizer(
    constraint_groups=cmp.all_constraints,
    primal_optimizers=primal_optimizer,
    dual_optimizers=dual_optimizer,
)

state_history = {}
for i in range(3500):
    cmp_state, lagrangian_store = cooper_optimizer.roll(
        compute_cmp_state_fn=lambda: cmp.compute_cmp_state(log_probs), return_multipliers=True
    )

    state_history[i] = {
        "loss": cmp_state.loss.item(),
        "multipliers": deepcopy(torch.stack([_.multiplier().data for _ in cmp.all_constraints])),
        "violation": deepcopy(torch.stack([_.state.violation.data for _ in cmp.all_constraints])),
    }

# Theoretical solution
optim_prob = torch.tensor([0.05435, 0.07877, 0.1142, 0.1654, 0.2398, 0.3475])
optim_neg_entropy = torch.sum(optim_prob * torch.log(optim_prob))

# Generate plots
iters, loss_hist, multipliers_hist, violation_hist = zip(
    *[(k, v["loss"], v["multipliers"], v["violation"]) for k, v in state_history.items()]
)

_, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 4))

ax0.plot(iters, torch.stack(multipliers_hist).squeeze().cpu())
ax0.set_title("Multipliers")

ax1.plot(iters, torch.stack(violation_hist).squeeze().cpu())
# Show that defect remains below/at zero
ax1.axhline(0.0, c="gray", alpha=0.35)
ax1.set_title("Defects")

ax2.plot(iters, loss_hist)
# Show optimal entropy is achieved
ax2.axhline(optim_neg_entropy, c="gray", alpha=0.35)
ax2.set_title("Objective")

plt.show()
