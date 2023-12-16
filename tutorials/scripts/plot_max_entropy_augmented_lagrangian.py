r"""
Finding a discrete maximum entropy distribution
===============================================

Here we consider a simple convex optimization problem to illustrate how to use
**Cooper**. This example is inspired by `this StackExchange question
<https://datascience.stackexchange.com/questions/107366/how-do-you-solve-strictly-constrained-optimization-problems-with-pytorch>`_\:

*I am trying to solve the following problem using Pytorch: given a 6-sided die
whose average roll is known to be 4.5, what is the maximum entropy distribution
for the faces?*

This tutorial shows how to use the Augmented Lagrangian  in **Cooper**.
"""

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from style_utils import *

import cooper

torch.manual_seed(0)
np.random.seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MaximumEntropy(cooper.ConstrainedMinimizationProblem):
    def __init__(self, target_mean: float) -> None:
        self.target_mean = target_mean

        default_multiplier_kwargs = {"constraint_type": cooper.ConstraintType.EQUALITY, "device": DEVICE}
        mean_multiplier = cooper.multipliers.DenseMultiplier(**default_multiplier_kwargs, num_constraints=1)
        mean_penalty_coefficient = cooper.multipliers.DensePenaltyCoefficient(torch.tensor(1.0, device=DEVICE))
        sum_multiplier = cooper.multipliers.DenseMultiplier(**default_multiplier_kwargs, num_constraints=1)

        self.mean_constraint = cooper.ConstraintGroup(
            constraint_type=cooper.ConstraintType.EQUALITY,
            formulation_type=cooper.FormulationType.AUGMENTED_LAGRANGIAN,
            multiplier=mean_multiplier,
            penalty_coefficient=mean_penalty_coefficient,
            formulation_kwargs={"penalty_growth_factor": 1.001},
        )
        self.sum_constraint = cooper.ConstraintGroup(
            constraint_type=cooper.ConstraintType.EQUALITY,
            formulation_type=cooper.FormulationType.LAGRANGIAN,
            multiplier=sum_multiplier,
        )

        self.multipliers = {"mean": mean_multiplier, "sum": sum_multiplier}
        self.penalty_coefficients = {"mean": mean_penalty_coefficient}
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
# For the Augmented Lagrangian, we need to configure the dual optimizer to SGD(lr=1.0)
dual_optimizer = torch.optim.SGD(dual_parameters, lr=1.0, maximize=True)

cooper_optimizer = cooper.optim.AugmentedLagrangianDualPrimalOptimizer(
    primal_optimizers=primal_optimizer, dual_optimizers=dual_optimizer, multipliers=cmp.multipliers.values()
)


state_history = {}
for i in range(3000):
    cmp_state, lagrangian_store = cooper_optimizer.roll(compute_cmp_state_fn=lambda: cmp.compute_cmp_state(log_probs))

    observed_violations = [constraint_state.violation.data for _, constraint_state in cmp_state.observed_constraints]
    observed_multipliers = [multiplier().data for multiplier in cmp.multipliers.values()]
    observed_penalty_coefficients = [pc().data for pc in cmp.penalty_coefficients.values()]

    state_history[i] = {
        "loss": -cmp_state.loss.item(),
        "multipliers": deepcopy(torch.stack(observed_multipliers)),
        "violation": deepcopy(torch.stack(observed_violations)),
        "penalty_coefficients": deepcopy(torch.stack(observed_penalty_coefficients)),
    }


# Theoretical solution
optimal_prob = torch.tensor([0.05435, 0.07877, 0.1142, 0.1654, 0.2398, 0.3475])
optimal_entropy = -torch.sum(optimal_prob * torch.log(optimal_prob))

# Generate plots
iters, loss_hist, multipliers_hist, violation_hist, penalty_hist = zip(
    *[(k, v["loss"], v["multipliers"], v["violation"], v["penalty_coefficients"]) for k, v in state_history.items()]
)

_, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(24, 4))

ax0.plot(iters, torch.stack(multipliers_hist).squeeze().cpu())
ax0.axhline(0.0, c="gray", alpha=0.35)
ax0.set_title("Multipliers")

ax1.plot(iters, torch.stack(penalty_hist).squeeze().cpu())
ax1.axhline(0.0, c="gray", alpha=0.35)
ax1.set_title("Penalty Coefficients")

ax2.plot(iters, torch.stack(violation_hist).squeeze().cpu(), label=["Sum Constraint", "Mean Constraint"])
ax2.legend()
# Show that defect remains below/at zero
ax2.axhline(0.0, c="gray", alpha=0.35)
ax2.set_title("Constraint Violations")

ax3.plot(iters, loss_hist)
# Show optimal entropy is achieved
ax3.axhline(optimal_entropy, c="gray", alpha=0.35, linestyle="dashed")
ax3.set_title("Objective")

plt.show()
