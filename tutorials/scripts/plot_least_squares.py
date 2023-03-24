from copy import deepcopy
from typing import Tuple

import random

import cooper
import matplotlib.pyplot as plt
import numpy as np
import torch

from cooper import CMPState, ConstraintGroup, ConstraintState
from cooper.optim import SimultaneousConstrainedOptimizer

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, BatchSampler


torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinearConstraintSystem(Dataset):
    def __init__(self, A: torch.Tensor, b: torch.Tensor):
        self.A = A
        self.b = b

    def __len__(self):
        return self.A.shape[0]

    def __getitem__(self, index: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index = torch.tensor(index, device=DEVICE)
        return self.A[index], self.b[index], index


class LeastSquares(cooper.ConstrainedMinimizationProblem):
    def __init__(self, eq_group: ConstraintGroup) -> None:
        self.eq_group = eq_group
        super().__init__()

    def compute_cmp_state(self, x: torch.Tensor, A: torch.Tensor, b: torch.Tensor, indices: torch.Tensor) -> CMPState:

        loss = torch.linalg.vector_norm(x) ** 2
        violations = torch.matmul(A, x) - b
        self.eq_group.state = ConstraintState(violation=violations, constraint_features=indices)

        return CMPState(loss=loss, observed_constraints=[self.eq_group])


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


n_vars = 4
n_eqs = 4
batch_size = 4
primal_lr = 1e-3
dual_lr = 1e-7
n_epochs = 1000

# Randomly generate a linear system and create a dataset with it
A = torch.rand(size=(n_eqs, n_vars), device=DEVICE)
b = torch.rand(size=(n_eqs, 1), device=DEVICE)
linear_system = LinearConstraintSystem(A, b)

# Find the optimal solution by solving the linear system with PyTorch
x_optim = torch.linalg.lstsq(A, b).solution

# Check that the solution is correct
assert torch.allclose(torch.matmul(A, x_optim), b, atol=1e-5)

# Create a dataloader that samples uniformly with replacement. Pass a generator and a
# worker_init_fn to ensure reproducibility
g = torch.Generator()
g.manual_seed(0)

# Create a random sampler and a batch sampler to sample the constraints uniformly with replacement
random_sampler = RandomSampler(linear_system, replacement=True)  # , generator=g)
batch_sampler = BatchSampler(random_sampler, batch_size=batch_size, drop_last=False)

# Create the dataloader
dataloader_kwargs = {
    "sampler": batch_sampler,
    "worker_init_fn": seed_worker,
    "batch_size": None,  # Set batch_size to None to avoid additional dimension in the
    # output of the dataloader. The batch size is already specified
    # in the sampler so there is no need to specify it again.
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

primal_optimizer = torch.optim.SGD([x], lr=primal_lr, momentum=0.7)
dual_optimizer = torch.optim.SGD(eq_group.multiplier.parameters(), lr=dual_lr)  # , momentum=0.7)

optimizer = SimultaneousConstrainedOptimizer(
    constraint_groups=eq_group,
    primal_optimizers=primal_optimizer,
    dual_optimizers=dual_optimizer,
)

# Run the optimization process
state_history = {}
for i in range(n_epochs):

    # Create empty tensor to accumulate the violation of the observed constraints
    acumulated_violation = torch.zeros_like(b)

    for sampled_A, sampled_b, indices in constraint_loader:
        optimizer.zero_grad()
        cmp_state = cmp.compute_cmp_state(x, sampled_A, sampled_b, indices)
        _ = cmp_state.populate_lagrangian()
        cmp_state.backward()
        optimizer.step()

        # Accumulate the violation of the observed constraints
        acumulated_violation[indices] += cmp_state.observed_constraints[0].state.violation[indices]

    if i % 20 == 0:
        state_history[i] = {
            "loss": cmp_state.loss.item(),
            "multipliers": deepcopy(eq_group.multiplier.weight.detach().data),
            "x": deepcopy(x.detach().data),
            "x_dif": deepcopy((x - x_optim).detach().data),
            "violation": acumulated_violation.detach().data,
        }

    if i % 100 == 0:
        print(f"Epoch {i} - loss: {cmp_state.loss.item():.4f}")


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
    print(f"PLOT {title}")
    ax.set_title(title)

plt.savefig("least_squares.png")
# plt.show()
