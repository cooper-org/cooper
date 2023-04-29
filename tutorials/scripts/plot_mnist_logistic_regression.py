"""
Training a logistic regression classifier on MNIST under a norm constraint
=====================================================================================

Here we consider a simple convex constrained optimization problem that involves
training a Logistic Regression clasifier on the MNIST dataset. The model is
constrained so that the squared L2 norm of its parameters is less than 1.

This example illustrates how **Cooper** integrates with:
    -  constructing a ``cooper.LagrangianFormulation`` and a ``cooper.SimultaneousOptimizer``
    -  models defined using a ``torch.nn.Module``,
    - CUDA acceleration,
    - typical machine learning training loops,
    - extracting the value of the Lagrange multipliers from a ``cooper.LagrangianFormulation``.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from style_utils import *
from torchvision import datasets, transforms

import cooper
from cooper import CMPState, ConstraintGroup, ConstraintState, ConstraintType
from cooper.optim import SimultaneousOptimizer

np.random.seed(0)
torch.manual_seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data", train=True, download=True, transform=data_transforms),
    batch_size=256,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

loss_fn = torch.nn.CrossEntropyLoss()

# Create a Logistic Regression model
model = torch.nn.Linear(in_features=28 * 28, out_features=10, bias=True)
model = model.to(DEVICE)

primal_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)

# Define the constraint group for the norm constraint
ineq_group = ConstraintGroup(
    constraint_type=ConstraintType.INEQUALITY, multiplier_kwargs={"shape": 1, "device": DEVICE}
)

# Instantiate Pytorch optimizer class for the dual variables
dual_optimizer = torch.optim.SGD(ineq_group.multiplier.parameters(), lr=1e-3)

cooper_optimizer = SimultaneousOptimizer(
    primal_optimizers=primal_optimizer, dual_optimizers=dual_optimizer, constraint_groups=ineq_group
)

all_metrics = {"batch_ix": [], "train_loss": [], "train_acc": [], "ineq_multiplier": [], "ineq_defect": []}

batch_ix = 0

for epoch_num in range(7):
    for inputs, targets in train_loader:
        batch_ix += 1

        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

        logits = model.forward(inputs.view(inputs.shape[0], -1))
        loss = loss_fn(logits, targets)
        accuracy = (logits.argmax(dim=1) == targets).float().mean()

        sq_l2_norm = model.weight.pow(2).sum() + model.bias.pow(2).sum()
        # Constraint defects use convention “g - \epsilon ≤ 0”
        constraint_defect = sq_l2_norm - 1.0
        ineq_group.state = ConstraintState(violation=constraint_defect)

        # Create a CMPState object, which contains the loss and observed constraints
        cmp_state = CMPState(loss=loss, observed_constraints=[ineq_group])

        cooper_optimizer.zero_grad()
        lagrangian_store = cmp_state.populate_lagrangian()
        cmp_state.backward()
        cooper_optimizer.step()

        # Extract the value of the Lagrange multiplier associated with the constraint
        multiplier_value = ineq_group.multiplier()

        if batch_ix % 3 == 0:
            all_metrics["batch_ix"].append(batch_ix)
            all_metrics["train_loss"].append(loss.item())
            all_metrics["train_acc"].append(accuracy.item())
            all_metrics["ineq_multiplier"].append(multiplier_value.item())
            all_metrics["ineq_defect"].append(constraint_defect.item())

fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=4, sharex=True, figsize=(18, 4))

ax0.plot(all_metrics["batch_ix"], all_metrics["train_loss"])
ax0.set_xlabel("Batch")
ax0.set_title("Training Loss")

ax1.plot(all_metrics["batch_ix"], all_metrics["train_acc"])
ax1.set_xlabel("Batch")
ax1.set_title("Training Acc")

ax2.plot(all_metrics["batch_ix"], np.stack(all_metrics["ineq_multiplier"]))
ax2.set_xlabel("Batch")
ax2.set_title("Inequality Multiplier")

ax3.plot(all_metrics["batch_ix"], np.stack(all_metrics["ineq_defect"]))
# Show that defect converges close to zero
ax3.axhline(0.0, c="gray", alpha=0.5)
ax3.set_xlabel("Batch")
ax3.set_title("Inequality Defect")

plt.show()
