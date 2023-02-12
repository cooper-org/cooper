"""
Training a logistic regression classifier on MNIST under a norm constraint
=====================================================================================

Here we consider a simple convex constrained optimization problem that involves
training a Logistic Regression clasifier on the MNIST dataset. The model is
constrained so that the squared L2 norm of its parameters is less than 1.

This example illustrates how **Cooper** integrates with:
    -  constructing a ``cooper.LagrangianFormulation`` and a ``cooper.SimultaneousConstrainedOptimizer``
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

np.random.seed(0)
torch.manual_seed(0)

data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data", train=True, download=True, transform=data_transforms),
    batch_size=256,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
)

loss_fn = torch.nn.CrossEntropyLoss()

# Create a Logistic Regression model
model = torch.nn.Linear(in_features=28 * 28, out_features=10, bias=True)
if torch.cuda.is_available():
    model = model.cuda()
primal_optimizer = torch.optim.Adagrad(model.parameters(), lr=5e-3)

# Create a Cooper formulation, and pick a Pytorch optimizer class for the dual variables
formulation = cooper.LagrangianFormulation()
dual_optimizer = cooper.optim.partial_optimizer(torch.optim.SGD, lr=1e-3)

# Create a ConstrainedOptimizer for performing simultaneous updates based on the
# formulation, and the selected primal and dual optimizers.
cooper_optimizer = cooper.SimultaneousConstrainedOptimizer(formulation, primal_optimizer, dual_optimizer)

all_metrics = {
    "batch_ix": [],
    "train_loss": [],
    "train_acc": [],
    "ineq_multiplier": [],
    "ineq_defect": [],
}

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

        # Create a CMPState object, which contains the loss and constraint defect
        cmp_state = cooper.CMPState(loss=loss, ineq_defect=constraint_defect)

        cooper_optimizer.zero_grad()
        lagrangian = formulation.compute_lagrangian(pre_computed_state=cmp_state)
        formulation.backward(lagrangian)
        cooper_optimizer.step()

        # Extract the value of the Lagrange multiplier associated with the constraint
        # The dual variables are stored and updated internally by Cooper
        lag_multiplier, _ = formulation.state()

        if batch_ix % 3 == 0:
            all_metrics["batch_ix"].append(batch_ix)
            all_metrics["train_loss"].append(loss.item())
            all_metrics["train_acc"].append(accuracy.item())
            all_metrics["ineq_multiplier"].append(lag_multiplier.item())
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
