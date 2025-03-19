# FAQ

TODO: emojis?

How can I tell if Cooper found a good solution?
  As a reference, consider the solution of the unconstrained problem, which is a lower bound on the solution to the constrained problem
  Nuance with the fact that you may not actually solve the problem in the nonconvex case
Primal optimization pipeline
  Tune with unconstrained
How to choose dual lr
  1e-3 to start
  If dual lr is Larger, pushing for feasibility faster.
  Relationship between mini-batch size, and the relative frequency of multiplier updates.
Noise
  What is noise? Constraints are estimated stochastically
  Also makes it tricky to determine if you are feasible.
  Difficult to achieve feasibility
  Consider evaluating the constraints at the epoch level/averaging out constraints
  Increase batch size
  Variance reduction


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
  <summary style="font-size: 1.2rem;">
    Where can I get help with <b>Cooper</b>?
  </summary>
  <div style="margin-left: 20px;">
    You can ask questions and get help on our <a href="https://discord.gg/Aq5PjH8m6E">Discord server</a>.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    Where can I learn more about constrained optimization?
  </summary>
  <div style="margin-left: 20px;">
    You can find more on convex constrained optimization in <a href="https://web.stanford.edu/~boyd/cvxbook/">Convex Optimization</a> by Boyd and Vandenberghe.
    For non-convex constrained optimization, you can check out <a href="http://athenasc.com/nonlinbook.html">Nonlinear Programming</a> by Bertsekas.
  </div>
</details>

### Formulations

<details>
  <summary style="font-size: 1.2rem;">
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
  <summary style="font-size: 1.2rem;">
    What is a good configuration for the primal optimizer?
  </summary>
  <div style="margin-left: 20px;">
    You can use whichever optimizer you prefer for your task, e.g., SGD, Adam, ...
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    What is a good configuration for the dual optimizer?
  </summary>
  <div style="margin-left: 20px;">
    For the dual optimizer, we recommend starting with SGD. If the dual learning rate is difficult to tune or if the Lagrange multipliers present oscillations, we recommend using <a href="TODO">nuPI</a>.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    Which <b>Cooper</b> optimizer should I use?
  </summary>
  <div style="margin-left: 20px;">
    <b>Cooper</b> provides a range of CooperOptimizers to choose from. The <b>AlternatingDualPrimalOptimizer</b> is a good starting point. For details, <a href=https://cooper.readthedocs.io/en/latest/optim.html>see</a>.
  </div>
</details>

### Debugging and troubleshooting

<details>
    <summary style="font-size: 1.2rem;">
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
  <summary style="font-size: 1.2rem;">
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
  <summary style="font-size: 1.2rem;">
    Why is my objective function increasing? 😟
  </summary>
  <div style="margin-left: 20px;">
    <b>It is common for the loss to temporarily increase when solving constrained optimization problems</b>, particularly when constraints are violated. This occurs because the drive for feasibility can conflict with minimizing the loss, causing a brief increase in the objective. However, as the optimization progresses, the loss should stabilize and potentially decrease, as the variables strike a balance between ensuring feasibility and achieving optimality.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    How can I tell if <b>Cooper</b> found a "good" solution?
  </summary>
  <div style="margin-left: 20px;">
  Consider the solution to the unconstrained version of the problem (with the same objective). This solution provides a lower bound for the constrained problem, as it can optimize the objective without needing to satisfy the constraints. Thus, <b>if Cooper's solution to the constrained problem is close to the unconstrained solution, it is likely a good one</b>.

  However, for nonconvex problems, the solutions found for either problem may not be globally optimal, making such assessments more challenging.

  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    What should I log to monitor the progress of my optimization?
  </summary>
  <div style="margin-left: 20px;">
    Log the loss, constraint violations, multiplier values, and the Lagrangian.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    What do typical multiplier dynamics look like?
  </summary>
  <div style="margin-left: 20px;">
    Answer here. Complementary slackness.
  </div>
</details>

**What should I do if my Lagrange multipliers diverge?**
> * Start by ensuring that your problem is feasible: for infeasible problems, the optimal Lagrange multipliers are infinite.
> * Normally, the growth in the Lagrange multipliers (due to the accumulation of the violation) is accompanied by a "response" from the primal parameters moving towards feasibility. A lack of primal response could be due to the primal learning rate being too low.
> * Having tuned the primal learning rate, a lack of primal response could indicate (i) that your problem is infeasible or (ii) that the constraint gradients are vanishing (impeding movement towards feasibility). In situation (ii), you may attempt reformulating the constraints to avoid the vanishing gradient.

<details>
  <summary style="font-size: 1.2rem;">
    What should I do if my Lagrange multipliers oscillate too much?
  </summary>
  <div style="margin-left: 20px;">
    Answer here.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    What should I do if my Lagrange multipliers are too noisy?
  </summary>
  <div style="margin-left: 20px;">
    Answer here.
  </div>
</details>

### Computational considerations

<details>
  <summary style="font-size: 1.2rem;">
    Is <b>Cooper</b> computationally expensive?
  </summary>
  <div style="margin-left: 20px;">
    Answer here.
  </div>
</details>


<details>
  <summary style="font-size: 1.2rem;">
    Does <b>Cooper</b> support GPU acceleration?
  </summary>
  <div style="margin-left: 20px;">
    Answer here.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    Does <b>Cooper</b> support DDP execution?
  </summary>
  <div style="margin-left: 20px;">
    Answer here.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    Does <b>Cooper</b> support AMP?
  </summary>
  <div style="margin-left: 20px;">
    Answer here.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    What if my problem has a lot of constraints?
  </summary>
  <div style="margin-left: 20px;">
    Answer here. IndexedMultipliers, ImplicitMultipliers, etc.
  </div>

### Advanced topics


### Miscellaneous

<details>
  <summary style="font-size: 1.2rem;">
    How do I cite <b>Cooper</b>?
  </summary>
  <div style="margin-left: 20px;">
    Answer here.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    Is there a JAX version of <b>Cooper</b>?
  </summary>
  <div style="margin-left: 20px;">
    Answer here.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    Is there a TensorFlow version of <b>Cooper</b>?
  </summary>
  <div style="margin-left: 20px;">
    Answer here. TFCO is a good alternative.
  </div>
</details>
