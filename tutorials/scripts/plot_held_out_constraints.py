"""
Linear transformation between two vectors with minimal MSE
==========================================================

This example considers the problem of finding the projection matrix X that minimizes the
mean squared error between two vectors y and z. In particular, we are interested in a
matrix X such that the geometric mean of its singular values is equal to some constant.
Formally,

.. math::
    \min_{X}  \,\, \Vert Xy - z \Vert_2^2  \,\, s.t. \,\, \prod_{i=1}^n \sigma_i(X) = c

where :math:`X` is a matrix of size :math:`(m, n)`, :math:`y` and :math:`z` are vectors
of size :math:`(m, 1)` and :math:`(n, 1)` respectively, and :math:`c` is a constant.
Here, :math:`\sigma_i(X)` denotes the :math:`i`-th singular value of :math:`X`.

TODO: what is the theoretical solution to this problem?

We use this problem to illustrate one of the features of Cooper: the ability to update
the primal and dual variables at different frequencies. In particular, we consider the
case where the true constraint is observed only sporadically. This is useful in cases
where the constraint is expensive to compute, but where a surrogate can be computed
cheaply.

Calculating the geometric mean of the singular values of X is expensive since it
requires computing the SVD decomposition of X. However, the *arithmetic* mean of the
(squared) singular values of X can be computed cheaply: it corresponds to the trace of
:math:`X X^T`. This tutorial considers the arithmetic mean as a surrogate constraint for
the geometric mean.
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import style_utils
import torch
from style_utils import *

import cooper

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_vectors(dim_y: int, dim_z: int, seed: int = 0):
    """Create y and z such that Xy = z is true for a well-conditioned matrix X."""

    torch.manual_seed(seed=seed)

    # Create a random linear system with 1-singular values
    U, _, V = torch.linalg.svd(torch.randn(dim_z, dim_y))
    S = torch.eye(dim_z, dim_y)
    X = U @ S @ V.T

    y = torch.randn(dim_y, 1)
    y = y / torch.linalg.norm(y)

    z = X @ y
    z = z / torch.linalg.norm(z)

    y, z = y.to(DEVICE), z.to(DEVICE)

    return y, z


class MinErrorWithSingularValueConstraints(cooper.ConstrainedMinimizationProblem):
    """Find a matrix to minimize the error of a linear system with a constraint on the
    geometric mean its singular values."""

    def __init__(self, y: torch.Tensor, z: torch.Tensor, constraint_level: float = 1.0):
        self.y, self.z = y, z
        self.constraint_level = constraint_level

        # Creating a constraint group with a single constraint
        constraint_type = cooper.ConstraintType.EQUALITY
        self.multiplier = cooper.multipliers.DenseMultiplier(
            constraint_type=constraint_type, num_constraints=1, device=DEVICE
        )
        self.constraint = cooper.ConstraintGroup(
            constraint_type=constraint_type,
            formulation_type=cooper.FormulationType.LAGRANGIAN,
            multiplier=self.multiplier,
        )
        super().__init__()

    def loss_fn(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the MSE loss function for a given X."""
        return torch.linalg.norm(X @ self.y - self.z).pow(2) / 2

    def compute_cmp_state(self, X: torch.Tensor, do_compute_violation: bool = False) -> cooper.CMPState:
        """Compute the CMPState for a given X."""

        # Compute the objective
        objective = self.loss_fn(X)

        if do_compute_violation:
            # Compute the non-differentiable constraint to update the multipliers.
            # This is only done sporadically since the SVD decomposition is expensive.

            # Geometric mean of singular values of X *equal* to `constraint_level`.
            strict_violation = torch.linalg.svdvals(X).prod() - self.constraint_level

        # Surrogate constraint: *arithmetic* mean of singular values of X.
        # Since the surrogate is only used to compute gradients, there is no use to set
        # a constraint level.
        surrogate = torch.trace(X @ X.T)

        if not do_compute_violation:
            # `strict_violation` is not measured, so only X is updated (and not the
            # multipliers). We must specify `contributes_to_dual_update=False` to avoid
            # updating the multipliers based on the surrogate constraint.
            constraint_state = cooper.ConstraintState(
                violation=surrogate, contributes_to_primal_update=True, contributes_to_dual_update=False
            )
        else:
            # The true violation is set in `strict_violation` and the surrogate is set
            # in `surrogate`. The surrogate is used to compute updates for X, whereas
            # the strict violation is used to compute updates for the multipliers.

            # Note that it is not necessary to set `contributes_to_primal_update=True`
            # *and* `contributes_to_dual_update=True`, since the default value is True.
            # We set them here for clarity.
            constraint_state = cooper.ConstraintState(
                violation=surrogate,
                strict_violation=strict_violation,
                contributes_to_primal_update=True,
                contributes_to_dual_update=True,
            )

        return cooper.CMPState(loss=objective, observed_constraints=[(self.constraint, constraint_state)])


def run_experiment(dim_y, dim_z, constraint_level, max_iter, tolerance, freq_for_dual_update, primal_lr, dual_lr):
    y, z = create_vectors(dim_y=dim_y, dim_z=dim_z, seed=0)

    X = torch.randn(dim_z, dim_y)
    X = X / torch.linalg.norm(X, keepdim=True).detach()

    # Creating X as a tensor from scratch for it to be a leaf tensor
    X = torch.tensor(X, requires_grad=True, device=DEVICE)

    cmp = MinErrorWithSingularValueConstraints(y=y, z=z, constraint_level=constraint_level)
    primal_optimizer = torch.optim.SGD([X], lr=primal_lr)
    dual_optimizer = torch.optim.SGD(cmp.multiplier.parameters(), lr=dual_lr, maximize=True, foreach=False)
    cooper_optimizer = cooper.optim.AlternatingDualPrimalOptimizer(
        primal_optimizers=primal_optimizer, dual_optimizers=dual_optimizer, multipliers=cmp.multiplier
    )

    # Initial values of the loss, trace and geometric mean
    with torch.no_grad():
        state_history = dict(
            loss=[cmp.loss_fn(X).item()],
            trace=[torch.trace(X @ X.T).item()],
            geometric_mean=[torch.linalg.svdvals(X).prod().item()],
            multiplier_values=[cmp.multiplier.weight.item()],
        )

    for iter in range(max_iter):
        prev_X = X.clone().detach()

        if iter % freq_for_dual_update == 0:
            # Compute the True violation
            compute_cmp_state_fn = lambda: cmp.compute_cmp_state(X, do_compute_violation=True)
            lagrangian_store = cooper_optimizer.roll(compute_cmp_state_fn=compute_cmp_state_fn)
        else:
            compute_cmp_state_fn = lambda: cmp.compute_cmp_state(X, do_compute_violation=False)
            lagrangian_store = cooper_optimizer.roll(compute_cmp_state_fn=compute_cmp_state_fn)

        if prev_X.allclose(X, atol=tolerance):
            break

        with torch.no_grad():
            state_history["loss"].append(cmp.loss_fn(X).item())
            state_history["trace"].append(torch.trace(X @ X.T).item())
            state_history["geometric_mean"].append(torch.linalg.svdvals(X).prod().item())
            state_history["multiplier_values"].append(cmp.multiplier.weight.item())

    return state_history


def plot_results(state_history, constraint_level):
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))

    ax[0, 0].plot(state_history["loss"])
    ax[0, 0].set_ylabel("MSE Loss")
    ax[0, 0].set_yscale("log")

    ax[0, 1].plot(state_history["trace"])
    ax[0, 1].set_ylabel("Arithmetic mean")

    ax[1, 0].plot(state_history["geometric_mean"])
    ax[1, 0].set_ylabel("Geometric mean")
    # Horizontal line at `constraint_level`
    ax[1, 0].axhline(constraint_level, color="red", linestyle="--", alpha=0.3)

    ax[1, 1].plot(state_history["multiplier_values"])
    ax[1, 1].set_ylabel("Multiplier")

    for ax_ in ax.flatten():
        ax_.set_xlabel("Iteration")
        for line in ax_.get_lines():
            line.set_linewidth(2)

    plt.tight_layout()

    plt.show()


dim_y, dim_z = 4, 4
constraint_level = 2.0
primal_lr, dual_lr = 1e-2, 1e-2
freq_for_dual_update = 100
max_iter, tolerance = 100000, 1e-6


state_history = run_experiment(
    dim_y=dim_y,
    dim_z=dim_z,
    constraint_level=constraint_level,
    max_iter=max_iter,
    tolerance=tolerance,
    freq_for_dual_update=freq_for_dual_update,
    primal_lr=primal_lr,
    dual_lr=dual_lr,
)
plot_results(state_history, constraint_level)
