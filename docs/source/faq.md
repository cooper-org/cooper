# FAQ

::::{container} last-list-item-no-margin
:::{dropdown} What are common pitfalls I should avoid when implementing a {py:class}`CMP<cooper.cmp.ConstrainedMinimizationProblem>`?
- **Constraints convention:** Ensure your constraints comply with **Cooper**'s convention: $\vg(\vx) \leq \vzero$ for inequality constraints and $\vh(\vx) = \vzero$ for equality constraints. If you have a $\vg(\vx) \geq \vzero$ constraint, provide **Cooper** with $-\vg(\vx) \leq \vzero$.
>
- **Gradient propagation:** Ensure that the tensors corresponding to the loss and constraints have gradients. Avoid "creating new tensors" for packing multiple constraints into a single tensor, as this can disrupt the computational graph and gradient backpropagation. For example, instead of using `torch.tensor([g1, g2, ...])`, use `torch.cat([g1, g2, ...])`. You can use the {py:meth}`~cooper.ConstrainedMinimizationProblem.sanity_check_cmp_state` method to check that constraints have gradients.
>
- **Efficiency tip:** To improve efficiency, reuse as much of the computational graph as possible between the loss and constraints. For example, if both depend on the outputs of a neural network, perform a single forward pass and reuse the computed outputs for both the loss and constraints.
:::
::::

:::{dropdown} What types of problems can I solve with **Cooper**?
**Cooper** is applicable to any constrained optimization problem with a (autograd) differentiable objective and constraints, including:

- convex and non-convex problems,
- problems involving stochastic functions, where the loss and/or constraints depend on data (such as typical deep learning tasks).

While **Cooper** requires constraint differentiability, the [proxy constraints](constrained_optimization.md#non-differentiable-constraints) feature allows handling non-differentiable constraints by using differentiable surrogates.



**Cooper** is designed for **general non-convex problems**, making minimal assumptions about the objective and constraints. For **convex problems with additional structure**, we recommend using specialized solvers.
:::

:::{dropdown} Where can I get help with **Cooper**?

If you spot any bugs, please raise an [issue](https://github.com/cooper-org/cooper/issues).

You can ask questions and get help on the [Discussions page](https://github.com/cooper-org/cooper/discussions).

:::

:::{dropdown} Where can I learn more about constrained optimization?
You can find more on convex constrained optimization in [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) by Boyd and Vandenberghe.
For non-convex constrained optimization, you can check out [Nonlinear Programming](http://athenasc.com/nonlinbook.html) by Bertsekas.

For an introduction to constrained optimization in machine learning problems, we recommend [Constrained Optimization for Machine Learning: Algorithms and Applications](https://umontreal.scholaris.ca/items/6bbfce14-9225-4e23-a94e-504bcfb1298f) by Jose Gallego-Posada.

:::

### Formulations

::::{container} last-list-item-no-margin
:::{dropdown} What problem formulations does **Cooper** support?

**Cooper** transforms a constrained optimization problem into a min-max problem thanks to a choice of problem [formulation](formulations.md).

The following formulations are supported in **Cooper**:
- [Lagrangian Formulation.](formulations.md#lagrangian-formulations)
- [Quadratic Penalty Formulation.](formulations.md#quadratic-penalty-formulations)
- [Augmented Lagrangian Formulation.](formulations.md#augmented-lagrangian-formulations)
:::
::::

### Optimizers

:::{dropdown} What is a good configuration for the primal optimizer?
**Cooper** is compatible with any PyTorch optimizer for updating the primal variables.

A good rule of thumb for selecting the primal optimizer is to use the **same optimizer (and hyperparameters) as you would for the unconstrained version of your problem**. This way, you can focus on tuning the dual optimizer without needing to redesign the primal optimization scheme.
:::

::::{container} last-list-item-no-margin
:::{dropdown} What is a good configuration for the dual optimizer?

Typically, **gradient ascent** ({py:class}`torch.optim.SGD``(maximize=True)`) is a good starting point for the dual optimizer. This is a simple and intuitive choice, where multipliers accumulate constraint violations.

The min-max nature of the problems can cause oscillations in the multipliers. To mitigate this, consider using the [$\nu$PI optimizer](torch_optimizers.md#nupi), which is specifically designed to stabilize multiplier dynamics.

We highlight that **momentum has been reported to exacerbate oscillations** in multipliers and is generally detrimental when optimizing dual variables in constrained optimization {cite:p}`sohrabi2024nupi`.

:::
::::

:::{dropdown} How should I tune the dual learning rate?
A larger dual learning rate will push the problem toward feasibility faster, but it may also introduce **divergence** or **oscillations** in the multipliers or cause **numerical instability**.

When setting the dual learning rate, you should consider the frequency of the dual updates. For example, updating the multipliers at every step might require a smaller step-size.

Moreover, many formulations, such as the {py:class}`cooper.formulations.Lagrangian` formulation, involve adding a weighted combination of the constraints to the loss. **When there are many constraints present, the dual step-size should be adjusted (reduced)** to prevent the constraints from dominating the contribution of the loss.

As a general rule of thumb, we recommend **setting the dual learning rate an order of magnitude higher than the _primal_ learning rate, divided by the number of constraints**. This will typically allow the multipliers to push the primal parameters toward feasibility with a reasonable level of aggressiveness, while still allowing for progress in reducing the loss.
:::

:::{dropdown} What is a "constrained optimizer" in **Cooper**, and how is it different from a PyTorch optimizer?

TODO

:::


:::{dropdown} What constrained optimizer should I use?
**Cooper** provides a range of [constrained optimizers](optim.md#constrained-optimizers) to choose from. The {py:class}`~cooper.optim.constrained_optimizers.AlternatingDualPrimalOptimizer` is a good starting point. For details, see [Optim](optim.md).
:::

### Debugging and troubleshooting

::::{container} last-list-item-no-margin
:::{dropdown} What behavior should I expect when solving a problem with **Cooper**?
1.  **If the initial solution is feasible:**
    - The loss will decrease.
    - Multipliers will remain at 0.
    - However, reducing the loss may sometimes introduce constraint violations, leading to behavior similar to the _infeasible case_.
2.  **If the initial solution is infeasible:**
    - The loss will initially decrease while multipliers "warm up."
    - As multipliers increase in magnitude, constraint violations should reduce. However, this may lead the loss to temporarily rise.
    - Once feasibility is reached, the loss should begin decreasing again.
    - If a constraint violation switches signs (e.g., an inequality becomes strictly satisfied), the corresponding multiplier should decrease. For inequalities, it may reach 0.
    - Eventually, both the loss and constraint violations should stabilize at an equilibrium.
:::
::::

::::{container} last-list-item-no-margin
:::{dropdown} Why is my solution not becoming feasible?
1.  **Assess whether your problem has feasible points**. If this cannot be determined from examining the constraints, try solving a "relaxed" version of the problem, where you focus only on finding feasible solutions without minimizing the loss. If you cannot find a feasible solution, your problem may be infeasible.
2.  Once you've confirmed feasibility, monitor the model's progress toward it. **If the primal parameters are not moving sufficiently fast toward feasibility, try adjusting (increasing) the dual learning rate** to apply more pressure on achieving feasibility.
:::
::::

:::{dropdown} Why is my objective function increasing? 😟
**It is common for the loss to temporarily increase when solving constrained optimization problems**, particularly when constraints are violated. This occurs because the drive for feasibility can conflict with minimizing the loss, causing a brief increase in the objective. However, as the optimization progresses, the loss should stabilize and potentially decrease, as the variables strike a balance between ensuring feasibility and achieving optimality.
:::

:::{dropdown} How can I tell if **Cooper** found a "good" solution?
Consider the solution to the unconstrained version of the problem (with the same objective). This solution provides a lower bound for the constrained problem, as it can optimize the objective without needing to satisfy the constraints. Thus, **if Cooper's solution to the constrained problem is close to the unconstrained solution, it is likely a good one**.

However, for nonconvex problems, the solutions found for either problem may not be globally optimal, making such assessments more challenging.
:::

:::{dropdown} What should I log to monitor the progress of my optimization?
Log the loss, constraint violations, multiplier values, and the Lagrangian.
:::

::::{container} last-list-item-no-margin
:::{dropdown} What do typical multiplier dynamics look like?
- **Inequality constraints:**
    - If a constraint is violated, its corresponding Lagrange multiplier increases to penalize the violation.
    - If the constraint is strictly satisfied, the multiplier decreases, shifting focus toward minimizing the loss.
    - At convergence, the multipliers for strictly satisfied constraints should be zero, while those for violated constraints stabilize at a positive value.
- **Equality constraints:**
    - If the constraint is violated, the multiplier increases (or decreases).
    - The multiplier stabilizes once the constraint is satisfied.
:::
::::

::::{container} last-list-item-no-margin
:::{dropdown} What should I do if my Lagrange multipliers diverge?
1.  If your problem does not have any feasible solutions, the Lagrange multipliers may grow indefinitely.
2.  Typically, the growth of Lagrange multipliers is accompanied by a corresponding response from the primal parameters moving toward feasibility. If there is no response from the primal parameters, it could be due to the primal learning rate being too low.
3.  If the primal learning rate is properly tuned and there is still no response, this may indicate one of the following:
    - The problem is infeasible.
    - The constraint gradients are vanishing, impeding movement toward feasibility. In this case, you may try reformulating the constraints to avoid vanishing gradients.
:::
::::

::::{container} last-list-item-no-margin
:::{dropdown} What should I do if my Lagrange multipliers oscillate too much?
Oscillations in Lagrange multipliers are common due to the game structure of the Lagrangian formulation, where feasibility and optimality must be balanced.
However, if the oscillations become too severe, try the following:

1.  Decrease one or both of the primal and dual learning rates.
2.  Consider a different dual optimizer, such as [nuPI](torch_optimizers.md#nupi), which is designed to mitigate oscillations.
:::
::::

::::{container} last-list-item-no-margin
:::{dropdown} What should I do if my Lagrange multipliers are too noisy?
Stochastic constraint estimation, such as when constraints depend on training data and are estimated from mini-batches, can introduce noise in the Lagrange multipliers. This stochasticity makes it difficult to determine feasibility, as a constraint may be satisfied for some stochastic samples but not others, leading to erratic multiplier updates.
If you're experiencing noisy multipliers, consider these strategies:

1.  Evaluate constraints at the epoch level or average them across multiple epochs to smooth the estimates.
2.  Increase the batch size to reduce variance in constraint estimations.
3.  Use variance reduction techniques like SAGA or SVRG. These methods compute aggregate constraint measurements across mini-batches, providing more stable multiplier updates.
:::
::::

### Computational considerations

:::{dropdown} Is **Cooper** computationally expensive?
No, **Cooper** is computationally efficient, comparable to solving unconstrained minimization problems in PyTorch. The only additional cost is the storage and updating of the Lagrange multipliers, which is generally negligible in large-scale machine learning applications.
In terms of computation, **Cooper** does not require extra forward or backward passes compared to an unconstrained problem:

*   The forward pass computes the Lagrangian by evaluating both the loss and constraints. In many cases, such as when the constraint depends on the model's output, the computational graph for the loss and constraints is largely shared. Additionally, the Lagrangian is a linear combination of the loss and constraints, making it inexpensive to compute.
*   The backward pass backpropagates through the loss and constraints, which is already required when minimizing the loss. The gradient of the Lagrangian with respect to the multipliers corresponds to the constraint violations themselves, so no additional backward passes or gradients are needed to update the multipliers. Only the already evaluated constraint violations are used.


Additionally, **Cooper** takes advantage of PyTorch's autograd functionality and GPU acceleration for these operations.
In terms of storage, you need to store the value of each Lagrange multiplier (one per constraint). This storage requirement is typically negligible unless the number of constraints exceeds the model or data sizes.
:::

:::{dropdown} Does **Cooper** support GPU acceleration?
Yes, **Cooper** supports GPU acceleration through PyTorch.
:::

:::{dropdown} Does **Cooper** support distributed data parallel (DDP) execution?
Currently, **Cooper** does not support DDP execution, but we have plans to implement this in the future.
:::

:::{dropdown} Does **Cooper** support automatic mixed-precision (AMP) training?
We have not tested **Cooper** with AMP, so we cannot guarantee that it operates as expected.
:::

:::{dropdown} What if my problem has a lot of constraints?
If your problem involves a large number of constraints, you can use [IndexedMultipliers](multipliers.md#indexed-multipliers) or [ImplicitMultipliers](multipliers.md#implicit-multipliers). The former allows for efficient indexing of the multiplier object, while the latter avoids explicitly storing them by considering a parametric representation instead.
:::

### Miscellaneous

:::{container} reset-highlight-margin
:::{dropdown} How do I cite **Cooper**?
To cite **Cooper**, please cite [this paper](link-to-paper):
#TODO: Add link to paper

    @misc{gallegoPosada2025cooper,
        author={Gallego-Posada, Jose and Ramirez, Juan and Hashemizadeh, Meraj and Lacoste-Julien, Simon},
        title={{Cooper: A Library for Constrained Optimization in Deep Learning}},
        howpublished={\url{https://github.com/cooper-org/cooper}},
        year={2025}
    }
:::
::::

:::{dropdown} Is there a JAX version of **Cooper**?
Not at the moment, but we’d love to see a JAX version of Cooper!
:::

:::{dropdown} Is there a TensorFlow version of **Cooper**?
Not exactly, but you can use TensorFlow Constrained Optimization (TFCO) for similar functionality.
:::
