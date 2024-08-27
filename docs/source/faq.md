# FAQ

TODO: emojis?

<details>
  <summary style="font-size: 1.2rem;">
    What types of problems can I solve with <b>Cooper</b>?
  </summary>
  <div style="margin-left: 20px;">
    Answer here. For convex problems or problems with special structure, suggest other libraries.
  </div>
</details>

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
    Why is my problem not becoming feasible?
  </summary>
  <div style="margin-left: 20px;">
    There are several reasons why this might happen.
    <ul>
      <li>Check if the constraints are correctly implemented.</li>
      <li>Check if the Lagrange multipliers are being updated correctly.</li>
      <li>Check if the dual learning rate is too high.</li>
    </ul>
  </div>
</details>


<details>
  <summary style="font-size: 1.2rem;">
    Why is my objective function increasing? 😟
  </summary>
  <div style="margin-left: 20px;">
    There are several reasons why this might happen. But the most common one is that the dual learning rate is too high. Try reducing it.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    How can I tell if <b>Cooper</b> found a "good" solution?
  </summary>
  <div style="margin-left: 20px;">
    Check the constraint violations. If the constraints are satisfied, you have a good solution.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    What quantities should I log for sanity-checking?
  </summary>
  <div style="margin-left: 20px;">
    Log the loss, the constraint violations, the multiplier values, and the Lagrangian.
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

<details>
  <summary style="font-size: 1.2rem;">
    What should I do if my Lagrange multipliers diverge?
  </summary>
  <div style="margin-left: 20px;">
    You can try reducing the learning rates or using a different optimizer.
  </div>
</details>

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
