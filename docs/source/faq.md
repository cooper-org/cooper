# FAQ

Primal optimization pipeline
  Tune with unconstrained

How to choose dual lr
  1e-3 to start
  If dual lr is Larger, pushing for feasibility faster.
  Relationship between mini-batch size, and the relative frequency of multiplier updates.



**What are common pitfalls when implementing a CMP?**

> * Make sure your constraints comply with **Cooper**'s  convention $g(\boldsymbol{x}) \leq 0$ for inequality constraints and $h(x) = 0$ for equality constraints. If you have a greater than or equal constraint $g(\boldsymbol{x}) \geq 0$, you should provide **Cooper** with $-g(\boldsymbol{x}) \leq 0$.
>
> * Make sure that the tensors corresponding to the loss and constraints have gradients. Avoid "creating **new** tensors" for packing multiple constraints in a single tensor as this could block gradient backpropagation: do not use `torch.tensor([g1, g2, ...])`; instead, use `torch.cat([g1, g2, ...])`. You can use the {py:meth}`~cooper.ConstrainedMinimizationProblem.sanity_check_cmp_state` to check this.
>
> * For efficiency, we suggest reusing as much of the computational graph as possible between loss and the constraints. For example, if both depend on the outputs of a neural network, we recommend performing a single forward pass and reusing the computed outputs for both the loss and the constraints.

**What types of problems can I solve with <b>Cooper</b>?**
Answer here. For convex problems or problems with special structure, suggest other libraries.


If non convex
Or stochastic
Autograd differentiable objective and constraints (or non-differentiable constraints but with a surrogate)


<details>
  <summary style="font-size: 1.1rem;">
    Where can I get help with <b>Cooper</b>?
  </summary>
  <div style="margin-left: 20px;">
    You can ask questions and get help on our <a href="https://discord.gg/Aq5PjH8m6E">Discord server</a>.
  </div>
</details>

<details>
  <summary style="font-size: 1.1rem;">
    Where can I learn more about constrained optimization?
  </summary>
  <div style="margin-left: 20px;">
    You can find more on convex constrained optimization in <a href="https://web.stanford.edu/~boyd/cvxbook/">Convex Optimization</a> by Boyd and Vandenberghe.
    For non-convex constrained optimization, you can check out <a href="http://athenasc.com/nonlinbook.html">Nonlinear Programming</a> by Bertsekas.
  </div>
</details>

### Formulations

<details>
  <summary style="font-size: 1.1rem;">
    What problem formulations does <b>Cooper</b> support?
  </summary>
  <div style="margin-left: 20px;">
    <b>Cooper</b> supports the following formulations:
    <ul>
      <li><a href="https://cooper.readthedocs.io/en/latest/lagrangian_formulation.html#lagrangian-formulation">Lagrangian Formulation.</a></li>
      <li><a href="https://cooper.readthedocs.io/en/latest/lagrangian_formulation.html#augmented-lagrangian-formulation">Augmented Lagrangian Formulation.</a></li>
    </ul>
  </div>
</details>

### Optimizers

<details>
  <summary style="font-size: 1.1rem;">
    What is a good configuration for the primal optimizer?
  </summary>
  <div style="margin-left: 20px;">
    You can use whichever optimizer you prefer for your task, e.g., SGD, Adam, ...
  </div>
</details>

<details>
  <summary style="font-size: 1.1rem;">
    What is a good configuration for the dual optimizer?
  </summary>
  <div style="margin-left: 20px;">
    For the dual optimizer, we recommend starting with SGD. If the dual learning rate is difficult to tune or if the Lagrange multipliers present oscillations, we recommend using <a href="TODO">nuPI</a>.
  </div>
</details>

<details>
  <summary style="font-size: 1.1rem;">
    Which <b>Cooper</b> optimizer should I use?
  </summary>
  <div style="margin-left: 20px;">
    <b>Cooper</b> provides a range of CooperOptimizers to choose from. The <b>AlternatingDualPrimalOptimizer</b> is a good starting point. For details, <a href=https://cooper.readthedocs.io/en/latest/optim.html>see</a>.
  </div>
</details>

### Debugging and troubleshooting

<details>
    <summary style="font-size: 1.1rem;">
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
  <summary style="font-size: 1.1rem;">
    Why is my solution not becoming feasible?
  </summary>
  <div style="margin-left: 20px;">
    <ol>
      <li><b>Assess whether your problem has feasible points</b>. If this cannot be determined from examining the constraints, try solving a "relaxed" version of the problem, where you focus only on finding feasible solutions without minimizing the loss. If you cannot find a feasible solution, your problem may be infeasible.</li>
      <li>Once you've confirmed feasibility, monitor the model's progress toward it. <b>If the primal parameters are not moving sufficiently fast toward feasibility, try adjusting (increasing) the dual learning rate</b> to apply more pressure on achieving feasibility.</li>
    </ol>
  </div>
</details>

<details>
  <summary style="font-size: 1.1rem;">
    Why is my objective function increasing? 😟
  </summary>
  <div style="margin-left: 20px;">
    <b>It is common for the loss to temporarily increase when solving constrained optimization problems</b>, particularly when constraints are violated. This occurs because the drive for feasibility can conflict with minimizing the loss, causing a brief increase in the objective. However, as the optimization progresses, the loss should stabilize and potentially decrease, as the variables strike a balance between ensuring feasibility and achieving optimality.
  </div>
</details>

<details>
  <summary style="font-size: 1.1rem;">
    How can I tell if <b>Cooper</b> found a "good" solution?
  </summary>
  <div style="margin-left: 20px;">
  Consider the solution to the unconstrained version of the problem (with the same objective). This solution provides a lower bound for the constrained problem, as it can optimize the objective without needing to satisfy the constraints. Thus, <b>if Cooper's solution to the constrained problem is close to the unconstrained solution, it is likely a good one</b>.

  However, for nonconvex problems, the solutions found for either problem may not be globally optimal, making such assessments more challenging.

  </div>
</details>

<details>
  <summary style="font-size: 1.1rem;">
    What should I log to monitor the progress of my optimization?
  </summary>
  <div style="margin-left: 20px;">
    Log the loss, constraint violations, multiplier values, and the Lagrangian.
  </div>
</details>

<details>
  <summary style="font-size: 1.1rem;">
    What do typical multiplier dynamics look like?
  </summary>
  <div style="margin-left: 20px;">
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
  <summary style="font-size: 1.1rem;">
    What should I do if my Lagrange multipliers diverge?
  </summary>
  <div style="margin-left: 20px;">
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
  <summary style="font-size: 1.1rem;">
    What should I do if my Lagrange multipliers oscillate too much?
  </summary>
  <div style="margin-left: 20px;">
    Oscillations in Lagrange multipliers are common due to the game structure of the Lagrangian formulation, where feasibility and optimality must be balanced. However, if the oscillations become too severe, try the following:
    <ol>
      <li>Decrease one or both of the primal and dual learning rates.</li>
      <li>Consider a different dual optimizer, such as <a href="https://cooper.readthedocs.io/en/latest/torch_optimizers.html#nupi">nuPI</a>, which is designed to mitigate oscillations.</li>
    </ol>
  </div>
</details>

<details>
  <summary style="font-size: 1.1rem;">
    What should I do if my Lagrange multipliers are too noisy?
  </summary>
  <div style="margin-left: 20px;">
    Stochastic constraint estimation, such as when constraints depend on training data and are estimated from mini-batches, can introduce noise in the Lagrange multipliers. This stochasticity makes it difficult to determine feasibility, as a constraint may be satisfied for some stochastic samples but not others, leading to erratic multiplier updates.

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
  <summary style="font-size: 1.1rem;">
    Is <b>Cooper</b> computationally expensive?
  </summary>
  <div style="margin-left: 20px;">
    <b>Cooper</b> is computationally efficient:
    <ul>
      <li>It requires only a few additional forward and backward passes to compute the Lagrangian.</li>
      <li>It relies on PyTorch for automatic differentiation and GPU acceleration.</li>
    </ul>
  </div>
</details>


<details>
  <summary style="font-size: 1.1rem;">
    Does <b>Cooper</b> support GPU acceleration?
  </summary>
  <div style="margin-left: 20px;">
    Yes, <b>Cooper</b> supports GPU acceleration through PyTorch.
  </div>
</details>

<details>
  <summary style="font-size: 1.1rem;">
    Does <b>Cooper</b> support DDP execution?
  </summary>
  <div style="margin-left: 20px;">
    Currently, <b>Cooper</b> does not support DDP execution, but we have plans to implement this in the future.
  </div>
</details>

<details>
  <summary style="font-size: 1.1rem;">
    Does <b>Cooper</b> support AMP?
  </summary>
  <div style="margin-left: 20px;">
    We have not tested <b>Cooper</b> with AMP, so we cannot guarantee that it operates as expected.
  </div>
</details>

<details>
  <summary style="font-size: 1.1rem;">
    What if my problem has a lot of constraints?
  </summary>
  <div style="margin-left: 20px;">
    If your problem involves a large number of constraints, you can utilize <b>IndexedMultipliers</b> or <b>ImplicitMultipliers</b>. This approach allows you to model the multipliers with fewer parameters, making the problem more manageable and efficient to solve.
  </div>
</details>

### Miscellaneous

<details>
  <summary style="font-size: 1.1rem;">
    How do I cite <b>Cooper</b>?
  </summary>
  <div style="margin-left: 20px;">
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
  <summary style="font-size: 1.1rem;">
    Is there a JAX version of <b>Cooper</b>?
  </summary>
  <div style="margin-left: 20px;">
    Not at the moment, but we’d love to see a JAX version of Cooper!
  </div>
</details>

<details>
  <summary style="font-size: 1.1rem;">
    Is there a TensorFlow version of <b>Cooper</b>?
  </summary>
  <div style="margin-left: 20px;">
    Not exactly, but you can use TensorFlow Constrained Optimization (TFCO) for similar functionality.
  </div>
</details>
