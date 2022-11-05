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

import matplotlib.pyplot as plt
import numpy as np
import torch

import cooper
import tutorials.scripts.style_utils as style_utils

torch.manual_seed(0)
np.random.seed(0)


class MaximumEntropy(cooper.ConstrainedMinimizationProblem):
    def __init__(self, mean_constraint):
        self.mean_constraint = mean_constraint
        super().__init__()

    def closure(self, probs):
        # Verify domain of definition of the functions
        assert torch.all(probs >= 0)

        # Negative signed removed since we want to *maximize* the entropy
        neg_entropy = torch.sum(probs * torch.log(probs))

        # Entries of p >= 0 (equiv. -p <= 0)
        ineq_defect = -probs

        # Equality constraints for proper normalization and mean constraint
        mean = torch.sum(torch.tensor(range(1, len(probs) + 1)) * probs)
        eq_defect = torch.stack([torch.sum(probs) - 1, mean - self.mean_constraint])

        return cooper.CMPState(
            loss=neg_entropy, eq_defect=eq_defect, ineq_defect=ineq_defect
        )


# Define the problem and formulation
cmp = MaximumEntropy(mean_constraint=4.5)
formulation = cooper.LagrangianFormulation(cmp)

# Define the primal parameters and optimizer
rand_init = torch.rand(6)  # Use a 6-sided die
probs = torch.nn.Parameter(rand_init / sum(rand_init))
primal_optimizer = cooper.optim.ExtraSGD([probs], lr=3e-2, momentum=0.7)

# Define the dual optimizer. Note that this optimizer has NOT been fully instantiated
# yet. Cooper takes care of this, once it has initialized the formulation state.
dual_optimizer = cooper.optim.partial_optimizer(
    cooper.optim.ExtraSGD, lr=9e-3, momentum=0.7
)

# Wrap the formulation and both optimizers inside a SimultaneousConstrainedOptimizer
coop = cooper.ExtrapolationConstrainedOptimizer(
    formulation, primal_optimizer, dual_optimizer
)

state_history = cooper.StateLogger(save_metrics=["loss", "eq_defect", "eq_multipliers"])

# Here is the actual training loop
for iter_num in range(2000):
    coop.zero_grad()
    lagrangian = formulation.compute_lagrangian(cmp.closure, probs)
    formulation.backward(lagrangian)
    coop.step(cmp.closure, probs)

    # Store optimization metrics at each step
    partial_dict = {"params": copy.deepcopy(probs.data)}
    state_history.store_metrics(formulation, iter_num, partial_dict)

all_metrics = state_history.unpack_stored_metrics()

# Theoretical solution
optim_prob = torch.tensor([0.05435, 0.07877, 0.1142, 0.1654, 0.2398, 0.3475])
optim_neg_entropy = torch.sum(optim_prob * torch.log(optim_prob))

# Generate plots
fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(15, 3))

ax0.plot(all_metrics["iters"], np.stack(all_metrics["eq_multipliers"]))
ax0.set_title("Multipliers")

ax1.plot(all_metrics["iters"], np.stack(all_metrics["eq_defect"]), alpha=0.6)
# Show that defect remains below/at zero
ax1.axhline(0.0, c="gray", alpha=0.35)
ax1.set_title("Defects")

ax2.plot(all_metrics["iters"], all_metrics["loss"])
# Show optimal entropy is achieved
ax2.axhline(optim_neg_entropy, c="gray", alpha=0.35)
ax2.set_title("Objective")

plt.show()
