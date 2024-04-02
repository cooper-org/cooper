"""
Training a logistic regression classifier on MNIST under a norm constraint
=====================================================================================

Here we consider a simple convex constrained optimization problem that involves
training a Logistic Regression clasifier on the MNIST dataset. The model is
constrained so that the squared L2 norm of its parameters is less than 1.

This example illustrates how **Cooper** integrates with:`
    - constructing a``cooper.optim.SimultaneousOptimizer``
    - models defined using a ``torch.nn.Module``,
    - CUDA acceleration,
    - typical machine learning training loops,
"""

import matplotlib.pyplot as plt
import numpy as np
import style_utils
import torch
from torchvision import datasets, transforms

import cooper

style_utils.set_plot_style()
np.random.seed(0)
torch.manual_seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NormConstrainedLogisticRegression(cooper.ConstrainedMinimizationProblem):
    def __init__(self, epsilon: float = 1.0):
        super().__init__()
        self.epsilon = epsilon

        multiplier = cooper.multipliers.DenseMultiplier(
            constraint_type=cooper.ConstraintType.INEQUALITY, num_constraints=1, device=DEVICE
        )

        self.constraint = cooper.Constraint(
            constraint_type=cooper.ConstraintType.INEQUALITY,
            formulation_type=cooper.LagrangianFormulation,
            multiplier=multiplier,
        )

    def compute_cmp_state(self, model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> cooper.CMPState:
        logits = model(inputs.view(inputs.shape[0], -1))
        loss = torch.nn.functional.cross_entropy(logits, targets)
        accuracy = (logits.argmax(dim=1) == targets).float().mean()

        sq_l2_norm = model.weight.pow(2).sum() + model.bias.pow(2).sum()
        # Constraint violations use convention “g - \epsilon ≤ 0”
        constraint_state = cooper.ConstraintState(violation=sq_l2_norm - self.epsilon)

        # Create a CMPState object, which contains the loss and observed constraints
        return cooper.CMPState(
            loss=loss, observed_constraints=[(self.constraint, constraint_state)], misc={"accuracy": accuracy}
        )


data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data", train=True, download=True, transform=data_transforms),
    batch_size=256,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

# Create a Logistic Regression model
model = torch.nn.Linear(in_features=28 * 28, out_features=10, bias=True)
model = model.to(DEVICE)

# Instantiate the constrained optimization problem
cmp = NormConstrainedLogisticRegression()

# Instantiate Pytorch optimizer class for the primal variables
primal_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)

# Instantiate Pytorch optimizer class for the dual variables
dual_optimizer = torch.optim.SGD(cmp.constraint.multiplier.parameters(), lr=1e-3, maximize=True)

# Instantiate the Cooper optimizer
cooper_optimizer = cooper.optim.SimultaneousOptimizer(
    primal_optimizers=primal_optimizer, dual_optimizers=dual_optimizer, cmp=cmp, multipliers=cmp.constraint.multiplier
)


all_metrics = {"batch_ix": [], "train_loss": [], "train_acc": [], "multiplier_value": [], "constraint_violation": []}

batch_ix = 0

for epoch_num in range(7):
    for inputs, targets in train_loader:
        batch_ix += 1

        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)

        cmp_state, lagrangian_store = cooper_optimizer.roll(
            compute_cmp_state_kwargs=dict(model=model, inputs=inputs, targets=targets)
        )

        if batch_ix % 3 == 0:
            all_metrics["batch_ix"].append(batch_ix)
            all_metrics["train_loss"].append(cmp_state.loss.item())
            all_metrics["train_acc"].append(cmp_state.misc["accuracy"].item())
            all_metrics["multiplier_value"].append(cmp.constraint.multiplier().item())
            all_metrics["constraint_violation"].append(cmp_state.observed_constraints[0][1].violation.item())

fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=4, sharex=True, figsize=(18, 4))

ax0.plot(all_metrics["batch_ix"], all_metrics["train_loss"])
ax0.set_xlabel("Batch")
ax0.set_title("Training Loss")

ax1.plot(all_metrics["batch_ix"], all_metrics["train_acc"])
ax1.set_xlabel("Batch")
ax1.set_title("Training Acc")

ax2.plot(all_metrics["batch_ix"], np.stack(all_metrics["multiplier_value"]))
ax2.set_xlabel("Batch")
ax2.set_title("Inequality Multiplier")

ax3.plot(all_metrics["batch_ix"], np.stack(all_metrics["constraint_violation"]))
# Show that defect converges close to zero
ax3.axhline(0.0, c="gray", alpha=0.5)
ax3.set_xlabel("Batch")
ax3.set_title("Inequality Defect")

plt.show()
