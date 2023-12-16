r"""
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
import style_utils
import torch

import cooper

style_utils.set_plot_style()

torch.manual_seed(0)
np.random.seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MaximumEntropy(cooper.ConstrainedMinimizationProblem):
    def __init__(self, target_mean: float) -> None:
        self.target_mean = target_mean

        default_multiplier_kwargs = {"constraint_type": cooper.ConstraintType.EQUALITY, "device": DEVICE}
        default_cg_kwargs = {
            "constraint_type": cooper.ConstraintType.EQUALITY,
            "formulation_type": cooper.FormulationType.LAGRANGIAN,
        }
        mean_multiplier = cooper.multipliers.DenseMultiplier(**default_multiplier_kwargs, num_constraints=1)
        sum_multiplier = cooper.multipliers.DenseMultiplier(**default_multiplier_kwargs, num_constraints=1)
        self.mean_constraint = cooper.ConstraintGroup(**default_cg_kwargs, multiplier=mean_multiplier)
        self.sum_constraint = cooper.ConstraintGroup(**default_cg_kwargs, multiplier=sum_multiplier)

        self.multipliers = {"mean": mean_multiplier, "sum": sum_multiplier}
        self.all_constraints = [self.sum_constraint, self.mean_constraint]

        super().__init__()

    def compute_cmp_state(self, log_probs: torch.Tensor) -> cooper.CMPState:
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs)

        # Equality constraints for proper normalization and mean constraint
        mean = torch.sum(probs * torch.arange(1, len(probs) + 1, device=DEVICE))

        sum_constraint_state = cooper.ConstraintState(violation=torch.sum(probs) - 1)
        mean_constraint_state = cooper.ConstraintState(violation=mean - self.target_mean)

        observed_constraints = [
            (self.sum_constraint, sum_constraint_state),
            (self.mean_constraint, mean_constraint_state),
        ]

        # Flip loss sign since we want to *maximize* the entropy
        return cooper.CMPState(loss=-entropy, observed_constraints=observed_constraints)


# Define the problem with the constraint groups
cmp = MaximumEntropy(target_mean=4.5)

# Define the primal parameters and optimizer
log_probs = torch.nn.Parameter(torch.log(torch.ones(6, device=DEVICE) / 6))
primal_optimizer = torch.optim.SGD([log_probs], lr=3e-2)

# Define the dual optimizer
dual_parameters = []
[dual_parameters.extend(multiplier.parameters()) for multiplier in cmp.multipliers.values()]
dual_optimizer = cooper.optim.PI(dual_parameters, lr=1e-2, Kp=10, maximize=True)

cooper_optimizer = cooper.optim.SimultaneousOptimizer(
    primal_optimizers=primal_optimizer, dual_optimizers=dual_optimizer, multipliers=cmp.multipliers.values()
)

state_history = {}
for i in range(3000):
    cmp_state, lagrangian_store = cooper_optimizer.roll(compute_cmp_state_fn=lambda: cmp.compute_cmp_state(log_probs))

    observed_violations = [constraint_state.violation.data for _, constraint_state in cmp_state.observed_constraints]
    observed_multipliers = [multiplier().data for multiplier in cmp.multipliers.values()]
    state_history[i] = {
        "loss": -cmp_state.loss.item(),
        "multipliers": deepcopy(torch.stack(observed_multipliers)),
        "violation": deepcopy(torch.stack(observed_violations)),
    }

# Theoretical solution
optimal_prob = torch.tensor([0.05435, 0.07877, 0.1142, 0.1654, 0.2398, 0.3475])
optimal_entropy = -torch.sum(optimal_prob * torch.log(optimal_prob))

# Generate plots
iters, loss_hist, multipliers_hist, violation_hist = zip(
    *[(k, v["loss"], v["multipliers"], v["violation"]) for k, v in state_history.items()]
)

_, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 4))

ax0.plot(iters, torch.stack(multipliers_hist).squeeze().cpu())
ax0.axhline(0.0, c="gray", alpha=0.35)
ax0.set_title("Multipliers")

ax1.plot(iters, torch.stack(violation_hist).squeeze().cpu(), label=["Sum Constraint", "Mean Constraint"])
ax1.legend()
# Show that defect remains below/at zero
ax1.axhline(0.0, c="gray", alpha=0.35)
ax1.set_title("Constraint Violations")

ax2.plot(iters, loss_hist)
# Show optimal entropy is achieved
ax2.axhline(optimal_entropy, c="gray", alpha=0.35, linestyle="dashed")
ax2.set_title("Objective")

plt.show()
