# FAQ

<details>
  <summary>
    What are common pitfalls when implementing a CMP?
  </summary>
  <div>
    <ul>
        <li><b>Constraints convention:</b> Ensure your constraints comply with <b>Cooper</b>'s convention: $\boldsymbol{g}(\boldsymbol{x}) \leq \boldsymbol{0}$ for inequality constraints and $\boldsymbol{h}(\boldsymbol{x}) = \boldsymbol{0}$ for equality constraints. If you have a $\boldsymbol{g}(\boldsymbol{x}) \geq \boldsymbol{0}$ constraint, provide <b>Cooper</b> with $-g(\boldsymbol{x}) \leq \boldsymbol{0}$.</li>
        <li><b>Gradient propagation:</b> Ensure that the tensors corresponding to the loss and constraints have gradients. Avoid "creating new tensors" for packing multiple constraints in a single tensor, as this can disrupt the computational graph and gradient backpropagation. Instead of using `torch.tensor([g1, g2, ...])`, use `torch.cat([g1, g2, ...])`. You can use the {py:meth}`~cooper.ConstrainedMinimizationProblem.sanity_check_cmp_state` to check that constraints have gradients.</li>
        <li><b>Efficiency tip:</b> To improve efficiency, reuse as much of the computational graph as possible between the loss and constraints. For example, if both depend on the outputs of a neural network, perform a single forward pass and reuse the computed outputs for both the loss and constraints.</li>
    </ul>
  </div>
</details>



<details>
  <summary>
    What types of problems can I solve with <b>Cooper</b>?
  </summary>
  <div>
    <b>Cooper</b> can solve any constrained optimization problem with a (autograd) differentiable objective and constraints, including:
    <ul>
        <li><b>Convex and non-convex problems</b></li>
        <li><b>Stochastic problems</b> where constraints depend on training data</li>
    </ul>
    While <b>Cooper</b> requires constraint differentiability, the <a href="https://cooper.readthedocs.io/en/latest/multipliers.html#surrogate-constraints">proxy constraints</a> feature allows handling non-differentiable constraints by using a differentiable surrogate.
    <br>
    <b>Cooper</b> is designed for <b>general non-convex problems</b>, making minimal assumptions about the objective and constraints.
    For <b>convex problems with additional structure</b>, we recommend using specialized solvers.
  </div>
</details>

<details>
  <summary>
    Where can I get help with <b>Cooper</b>?
  </summary>
  <div>
    You can ask questions and get help on our <a href="https://discord.gg/Aq5PjH8m6E">Discord server</a>.
  </div>
</details>

<details style="margin-bottom: 24px;">
  <summary>
    Where can I learn more about constrained optimization?
  </summary>
  <div>
    You can find more on convex constrained optimization in <a href="https://web.stanford.edu/~boyd/cvxbook/">Convex Optimization</a> by Boyd and Vandenberghe.
    For non-convex constrained optimization, you can check out <a href="http://athenasc.com/nonlinbook.html">Nonlinear Programming</a> by Bertsekas.
  </div>
</details>

### Formulations

<details style="margin-bottom: 24px;">
  <summary>
    What problem formulations does <b>Cooper</b> support?
  </summary>
  <div>
    <b>Cooper</b> supports the following formulations:
    <ul>
      <li><a href="https://cooper.readthedocs.io/en/latest/formulations.html#lagrangian-formulations">Lagrangian Formulation.</a></li>
      <li><a href="https://cooper.readthedocs.io/en/latest/formulations.html#quadratic-penalty-formulations">Quadratic Penalty Formulation.</a></li>
      <li><a href="https://cooper.readthedocs.io/en/latest/formulations.html#augmented-lagrangian-formulations">Augmented Lagrangian Formulation.</a></li>
    </ul>
  </div>
</details>

### Optimizers

<details>
  <summary>
    What is a good configuration for the primal optimizer?
  </summary>
  <div>
    <b>Cooper</b> works with any PyTorch optimizer. We recommend using the same optimizer as in the unconstrained version of your problem to avoid redesigning the primal optimization scheme. This allows you to focus on tuning the dual optimizer.
  </div>
</details>

<details>
  <summary>
    What is a good configuration for the dual optimizer?
  </summary>
  <div>
  <ul>
    <li><b>Gradient ascent (SGD)</b> is the simplest and most intuitive choice, where multipliers accumulate constraint violations.</li>
    <li>However, it can cause <b>oscillations</b> in the multipliers. To mitigate this, consider using <a href="https://cooper.readthedocs.io/en/latest/torch_optimizers.html#nupi"><b>nuPI</b></a>, which is specifically designed to <b>stabilize multiplier dynamics</b>.</li>
    <li><b>Important Note:</b> Momentum can <b>exacerbate oscillations</b> in multipliers and is generally detrimental when optimizing dual variables in constrained optimization (<a href="https://arxiv.org/abs/2406.04558">Sohrabi et al., 2024</a>).</li>
  </ul>
  </div>
</details>

<details>
  <summary>
    How should I tune the dual learning rate?
  </summary>
  <div>
  <p>
      As a general rule of thumb, set the dual learning rate about one order of magnitude higher than the primal learning rate. This will typically allow the multipliers to push the primal parameters toward feasibility with a reasonable level of aggressiveness.
  </p>
  <p>
      Note that a larger dual learning rate will push the problem toward feasibility faster, but it may also introduce <b>divergence</b> or <b>oscillations</b> in the multipliers or cause <b>numerical instability</b>.
  </p>
  <p>
      When setting the dual learning rate, also consider the frequency of dual updates. For example, updating every step might require smaller steps, while updating every epoch might allow for larger steps.
  </p>
  </div>
</details>

<details style="margin-bottom: 24px;">
  <summary>
    Which <b>Cooper</b> constrained optimizer should I use?
  </summary>
  <div>
    <b>Cooper</b> provides a range of constrained optimizers to choose from. The <b>AlternatingDualPrimalOptimizer</b> is a good starting point. For details, see <a href=https://cooper.readthedocs.io/en/latest/optim.html>see</a>.
  </div>
</details>

### Debugging and troubleshooting

<details>
    <summary>
    What behavior should I expect when solving a problem with <b>Cooper</b>?</summary>
    <div>
        <ol>
            <li><b>If the initial solution is feasible:</b>
                <ul>
                    <li>The loss will decrease.</li>
                    <li>Multipliers will remain at 0.</li>
                    <li>However, reducing the loss may sometimes introduce constraint violations, leading to behavior similar to the <i>infeasible case</i>.</li>
                </ul>
            </li>
            <li><b>If the initial solution is infeasible:</b>
                <ul>
                    <li>The loss will initially decrease while multipliers "warm up."</li>
                    <li>As multipliers increase in magnitude, constraint violations should reduce. However, this may lead the loss to temporarily rise.</li>
                    <li>Once feasibility is reached, the loss should begin decreasing again.</li>
                    <li>If a constraint violation switches signs (e.g., an inequality becomes strictly satisfied), the corresponding multiplier should decrease. For inequalities, it may reach 0.</li>
                    <li>Eventually, both the loss and constraint violations should stabilize at an equilibrium.</li>
                </ul>
            </li>
        </ol>
    </div>
</details>


<details>
  <summary>
    Why is my solution not becoming feasible?
  </summary>
  <div>
    <ol>
      <li><b>Assess whether your problem has feasible points</b>. If this cannot be determined from examining the constraints, try solving a "relaxed" version of the problem, where you focus only on finding feasible solutions without minimizing the loss. If you cannot find a feasible solution, your problem may be infeasible.</li>
      <li>Once you've confirmed feasibility, monitor the model's progress toward it. <b>If the primal parameters are not moving sufficiently fast toward feasibility, try adjusting (increasing) the dual learning rate</b> to apply more pressure on achieving feasibility.</li>
    </ol>
  </div>
</details>

<details>
  <summary>
    Why is my objective function increasing? 😟
  </summary>
  <div>
    <b>It is common for the loss to temporarily increase when solving constrained optimization problems</b>, particularly when constraints are violated. This occurs because the drive for feasibility can conflict with minimizing the loss, causing a brief increase in the objective. However, as the optimization progresses, the loss should stabilize and potentially decrease, as the variables strike a balance between ensuring feasibility and achieving optimality.
  </div>
</details>

<details>
  <summary>
    How can I tell if <b>Cooper</b> found a "good" solution?
  </summary>
  <div>
  Consider the solution to the unconstrained version of the problem (with the same objective). This solution provides a lower bound for the constrained problem, as it can optimize the objective without needing to satisfy the constraints. Thus, <b>if Cooper's solution to the constrained problem is close to the unconstrained solution, it is likely a good one</b>.

  However, for nonconvex problems, the solutions found for either problem may not be globally optimal, making such assessments more challenging.

  </div>
</details>

<details>
  <summary>
    What should I log to monitor the progress of my optimization?
  </summary>
  <div>
    Log the loss, constraint violations, multiplier values, and the Lagrangian.
  </div>
</details>

<details>
  <summary>
    What do typical multiplier dynamics look like?
  </summary>
  <div>
    <ul>
      <li><b>Inequality constraints:</b>
        <ul>
          <li>If a constraint is violated, its corresponding Lagrange multiplier increases to penalize the violation.</li>
          <li>If the constraint is strictly satisfied, the multiplier decreases, shifting focus toward minimizing the loss.</li>
          <li>At convergence, the multipliers for strictly satisfied constraints should be zero, while those for violated constraints stabilize at a positive value.</li>
        </ul>
      </li>
      <li><b>Equality constraints:</b>
        <ul>
          <li>If the constraint is violated, the multiplier increases (or decreases).</li>
          <li>The multiplier stabilizes once the constraint is satisfied.</li>
        </ul>
      </li>
    </ul>
  </div>
</details>


<details>
  <summary>
    What should I do if my Lagrange multipliers diverge?
  </summary>
  <div>
    <ol>
      <li>If your problem does not have any feasible solutions, the Lagrange multipliers may grow indefinitely.</li>
      <li>Typically, the growth of Lagrange multipliers is accompanied by a corresponding response from the primal parameters moving toward feasibility. If there is no response from the primal parameters, it could be due to the primal learning rate being too low.</li>
      <li>If the primal learning rate is properly tuned and there is still no response, this may indicate one of the following:
        <ul>
          <li>The problem is infeasible.</li>
          <li>The constraint gradients are vanishing, impeding movement toward feasibility. In this case, you may try reformulating the constraints to avoid vanishing gradients.</li>
        </ul>
      </li>
    </ol>
  </div>
</details>

<details>
  <summary>
    What should I do if my Lagrange multipliers oscillate too much?
  </summary>
  <div>
    Oscillations in Lagrange multipliers are common due to the game structure of the Lagrangian formulation, where feasibility and optimality must be balanced.
    <br>
    However, if the oscillations become too severe, try the following:
    <ol>
      <li>Decrease one or both of the primal and dual learning rates.</li>
      <li>Consider a different dual optimizer, such as <a href="https://cooper.readthedocs.io/en/latest/torch_optimizers.html#nupi">nuPI</a>, which is designed to mitigate oscillations.</li>
    </ol>
  </div>
</details>

<details style="margin-bottom: 24px;">
  <summary>
    What should I do if my Lagrange multipliers are too noisy?
  </summary>
  <div>
    Stochastic constraint estimation, such as when constraints depend on training data and are estimated from mini-batches, can introduce noise in the Lagrange multipliers. This stochasticity makes it difficult to determine feasibility, as a constraint may be satisfied for some stochastic samples but not others, leading to erratic multiplier updates.
    <br>
    If you're experiencing noisy multipliers, consider these strategies:
    <ol>
      <li>Evaluate constraints at the epoch level or average them across multiple epochs to smooth the estimates.</li>
      <li>Increase the batch size to reduce variance in constraint estimations.</li>
      <li>Use variance reduction techniques like SAGA or SVRG. These methods compute aggregate constraint measurements across mini-batches, providing more stable multiplier updates.</li>
    </ol>
  </div>
</details>


### Computational considerations

<details>
  <summary>
    Is <b>Cooper</b> computationally expensive?
  </summary>
  <div>
    No, <b>Cooper</b> is computationally efficient, comparable to solving unconstrained minimization problems in PyTorch. The only additional cost is the storage and updating of the Lagrange multipliers, which is generally negligible in large-scale machine learning applications.
    <br>
    In terms of computation, <b>Cooper</b> does not require extra forward or backward passes compared to an unconstrained problem:
    <ul>
      <li>The forward pass computes the Lagrangian by evaluating both the loss and constraints. In many cases, such as when the constraint depends on the model's output, the computational graph for the loss and constraints is largely shared. Additionally, the Lagrangian is a linear combination of the loss and constraints, making it inexpensive to compute.</li>
      <li>The backward pass backpropagates through the loss and constraints, which is already required when minimizing the loss. The gradient of the Lagrangian with respect to the multipliers corresponds to the constraint violations themselves, so no additional backward passes or gradients are needed to update the multipliers. Only the already evaluated constraint violations are used.</li>
    </ul>
    <br>
    Additionally, <b>Cooper</b> takes advantage of PyTorch's autograd functionality and GPU acceleration for these operations.
    <br>
    In terms of storage, you need to store the value of each Lagrange multiplier (one per constraint). This storage requirement is typically negligible unless the number of constraints exceeds the model or data sizes.
  </div>
</details>

<details>
  <summary>
    Does <b>Cooper</b> support GPU acceleration?
  </summary>
  <div>
    Yes, <b>Cooper</b> supports GPU acceleration through PyTorch.
  </div>
</details>

<details>
  <summary>
    Does <b>Cooper</b> support DDP execution?
  </summary>
  <div>
    Currently, <b>Cooper</b> does not support DDP execution, but we have plans to implement this in the future.
  </div>
</details>

<details>
  <summary>
    Does <b>Cooper</b> support AMP?
  </summary>
  <div>
    We have not tested <b>Cooper</b> with AMP, so we cannot guarantee that it operates as expected.
  </div>
</details>

<details style="margin-bottom: 24px;">
  <summary>
    What if my problem has a lot of constraints?
  </summary>
  <div>
    If your problem involves a large number of constraints, you can use <a href="https://cooper.readthedocs.io/en/latest/multipliers.html#indexed-multipliers">IndexedMultipliers</a> or <a href="https://cooper.readthedocs.io/en/latest/multipliers.html#implicit-multipliers">ImplicitMultipliers</a>. The former allows for efficient indexing of the multiplier object, while the latter avoids explicitly storing them by considering a parametric representation instead.
  </div>
</details>

### Miscellaneous

<details>
  <summary>
    How do I cite <b>Cooper</b>?
  </summary>
  <div>
    To cite <b>Cooper</b>, please cite [this paper](link-to-paper):
    #TODO: Add link to paper

    @misc{gallegoPosada2025cooper,
        author={Gallego-Posada, Jose and Ramirez, Juan and Hashemizadeh, Meraj and Lacoste-Julien, Simon},
        title={{Cooper: A Library for Constrained Optimization in Deep Learning}},
        howpublished={\url{https://github.com/cooper-org/cooper}},
        year={2025}
    }

  </div>
</details>

<details>
  <summary>
    Is there a JAX version of <b>Cooper</b>?
  </summary>
  <div>
    Not at the moment, but we’d love to see a JAX version of Cooper!
  </div>
</details>

<details>
  <summary>
    Is there a TensorFlow version of <b>Cooper</b>?
  </summary>
  <div>
    Not exactly, but you can use TensorFlow Constrained Optimization (TFCO) for similar functionality.
  </div>
</details>
