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

import copy
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from style_utils import *

import cooper

from cooper import CMPState, ConstraintGroup, ConstraintState
from cooper.optim import SimultaneousConstrainedOptimizer

torch.manual_seed(0)
np.random.seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MaximumEntropy(cooper.ConstrainedMinimizationProblem):
    def __init__(self, mean_constraint: float, ineq_group: ConstraintGroup, eq_group: ConstraintGroup) -> None:
        self.mean_constraint = mean_constraint
        self.ineq_group = ineq_group
        self.eq_group = eq_group
        super().__init__()

    def compute_cmp_state(self, probs: torch.Tensor) -> CMPState:
        # Verify domain of definition of the functions
        assert torch.all(probs >= 0)

        # Loose negative sign since we want to *maximize* the entropy
        entropy = torch.sum(probs * torch.log(probs))

        # Entries of p >= 0 (equiv. to -p <= 0)
        self.ineq_group.state = ConstraintState(violation=-probs)

        # Equality constraints for proper normalization and mean constraint
        mean = torch.sum(torch.tensor(range(1, len(probs) + 1)) * probs)
        self.eq_group.state = ConstraintState(
            violation=torch.stack([torch.sum(probs) - 1, mean - self.mean_constraint])
        )

        return CMPState(loss=entropy, observed_constraints=[self.ineq_group, self.eq_group])


# Define the constraint groups
ineq_group = ConstraintGroup(constraint_type="ineq", shape=6, dtype=torch.float16, device=DEVICE)
eq_group = ConstraintGroup(constraint_type="eq", shape=2, dtype=torch.float16, device=DEVICE)

# Define the problem with the constraint groups
cmp = MaximumEntropy(mean_constraint=4.5, ineq_group=ineq_group, eq_group=eq_group)

# Define the primal parameters and optimizer
rand_init = torch.rand(6, device=DEVICE)  # Use a 6-sided die
probs = torch.nn.Parameter(rand_init / sum(rand_init))
primal_optimizer = torch.optim.SGD([probs], lr=3e-2, momentum=0.7)

# Define the dual optimizer
dual_optimizer = torch.optim.SGD(
    [ineq_group.multiplier.parameters(), eq_group.multiplier.parameters()], lr=1e-3, momentum=0.4
)

# Wrap the formulation and both optimizers inside a SimultaneousConstrainedOptimizer
optimizer = SimultaneousConstrainedOptimizer(
    constraint_groups=eq_group,
    primal_optimizers=primal_optimizer,
    dual_optimizers=dual_optimizer,
)

state_history = {}
for i in range(5000):
    optimizer.zero_grad()
    cmp_state = cmp.compute_cmp_state(probs)
    _ = cmp_state.populate_lagrangian()
    cmp_state.backward()
    optimizer.step()

    state_history[i] = {
        "loss": cmp_state.loss.item(),
        "multipliers": deepcopy(eq_group.multiplier.weight.data),
        "violation": deepcopy(cmp_state.observed_constraints[1].state.violation.data),
    }

# Theoretical solution
optim_prob = torch.tensor([0.05435, 0.07877, 0.1142, 0.1654, 0.2398, 0.3475])
optim_neg_entropy = torch.sum(optim_prob * torch.log(optim_prob))

# Generate plots
iters, loss_hist, multipliers_hist, violation_hist = zip(
    *[(k, v["loss"], v["multipliers"], v["violation"]) for k, v in state_history.items()]
)

hist_list = [violation_hist, multipliers_hist]
hist_names = ["violation", "multipliers"]

_, (ax0, ax1, ax2) = plt.subplots(1, len(hist_list), figsize=(20, 4))

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
