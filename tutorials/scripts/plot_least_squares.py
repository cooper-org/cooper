from copy import deepcopy
from typing import Tuple

import random

import cooper
import matplotlib.pyplot as plt
import numpy as np
import torch

from cooper import CMPState, ConstraintGroup, ConstraintState
from cooper.optim import SimultaneousConstrainedOptimizer, ConstrainedOptimizer

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler


torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class LinearConstraintSystem(Dataset):
    def __init__(self, A: torch.Tensor, b: torch.Tensor):
        self.A = A
        self.b = b

    def __len__(self):
        return self.A.shape[0]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.A[index], self.b[index], index


class LeastSquares(cooper.ConstrainedMinimizationProblem):
    def __init__(self, eq_group: ConstraintGroup) -> None:
        self.eq_group = eq_group
        super().__init__()

    def compute_cmp_state(self, x: torch.Tensor, A: torch.Tensor, b: torch.Tensor, indices: torch.Tensor) -> CMPState:

        loss = torch.linalg.vector_norm(x) ** 2
        violations = torch.mm(A, x) - b
        self.eq_group.state = ConstraintState(violation=violations, constraint_features=indices)

        return CMPState(loss=loss, observed_constraints=[self.eq_group])


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


n_vars = 100
n_eqs = n_vars - 1
batch_size = n_eqs
primal_lr = 1e-5
dual_lr = 1e-5
n_epochs = 100

# Randomly generate a linear system and create a dataset with it
A = torch.rand(size=(n_eqs, n_vars), device=DEVICE)
b = torch.rand(size=(n_eqs, 1), device=DEVICE)
linear_system = LinearConstraintSystem(A, b)

# Find the optimal solution by solving the linear system with PyTorch
x_optim = torch.linalg.lstsq(A, b).solution

# Create a dataloader that samples uniformly with replacement. Pass a generator and a
# worker_init_fn to ensure reproducibility
g = torch.Generator().manual_seed(0)
random_sampler = RandomSampler(linear_system, replacement=True, num_samples=batch_size, generator=g)
sampler_type = "sampler" if n_eqs == batch_size else "batch_sampler"
dataloader_kwargs = {
    sampler_type: random_sampler,
    "worker_init_fn": seed_worker,
}
constraint_loader = DataLoader(linear_system, **dataloader_kwargs)

# Create a constraint group for the equality constraints. We use a sparse constraint
# to be able to update the multipliers only with the observed constraints (i.e. the
# ones that are active in the current batch)
eq_group = ConstraintGroup(constraint_type="eq", shape=A.shape[0], dtype=torch.float32, device=DEVICE, is_sparse=True)

# Define the problem with the constraint group
cmp = LeastSquares(eq_group=eq_group)

# Randomly initialize the primal variable and instantiate the optimizers
x = torch.nn.Parameter(torch.rand(n_vars, 1, device=DEVICE))

primal_optimizer = torch.optim.SGD([x], lr=primal_lr)
dual_optimizer = torch.optim.SGD(eq_group.multiplier.parameters(), lr=dual_lr)

optimizer = SimultaneousConstrainedOptimizer(
    constraint_groups=eq_group,
    primal_optimizers=primal_optimizer,
    dual_optimizers=dual_optimizer,
)

# Run the optimization process
state_history = {}
for i in range(n_epochs):
    for sampled_A, sampled_b, indices in constraint_loader:
        indices = indices.to(device=DEVICE)
        optimizer.zero_grad()
        cmp_state = cmp.compute_cmp_state(x, sampled_A, sampled_b, indices)
        _ = cmp_state.populate_lagrangian()
        cmp_state.backward()
        optimizer.step()

    state_history[i] = {
        "loss": cmp_state.loss.item(),
        "multipliers": deepcopy(eq_group.multiplier.weight.data),
        "x": deepcopy(x.data),
        "x_dif": deepcopy((x - x_optim).data),
        "violation": deepcopy(cmp_state.observed_constraints[0].state.violation.data),
    }

    if i % 100 == 0:
        print(f"Iteration: {i}, Loss: {cmp_state.loss.item()}")

# Plot the results
iters, loss_hist, multipliers_hist, x_hist, x_dif_hist, violation_hist = zip(
    *[(k, v["loss"], v["multipliers"], v["x"], v["x_dif"], v["violation"]) for k, v in state_history.items()]
)

fig, ax = plt.subplots(1, 5, figsize=(20, 4), sharex=True)

hist_list = [multipliers_hist, x_hist, x_dif_hist, violation_hist]
hist_names = ["multipliers", "x", "x_dif", "violation"]

ax[0].plot(iters, loss_hist)
ax[0].set_title("loss")

for ax, hist, title in zip(ax[1:], hist_list, hist_names):
    ax.plot(iters, torch.stack(hist).squeeze().cpu())
    ax.set_title(title)
plt.show()
