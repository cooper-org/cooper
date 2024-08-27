# FAQ

TODO: subsections here?
TODO: emojis?

<details>
  <summary style="font-size: 1.2rem;">
    What types of problems can I solve with <b>Cooper</b>?
  </summary>
  <div style="margin-left: 20px;">
    Answer here. For convex problems or problems with special structure, suggest other libraries.
  </div>
</details>

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
    You can find more on convex constrained optimization in the book <a href="https://web.stanford.edu/~boyd/cvxbook/">Convex Optimization</a> by Boyd and Vandenberghe.
    For non-convex constrained optimization, you can check out the book <a href="http://athenasc.com/nonlinbook.html">Nonlinear Programming</a> by Bertsekas.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    What kind of problems can I solve with <b>Cooper</b>?
  </summary>
  <div style="margin-left: 20px;">
    <b>Cooper</b> is designed to solve constrained optimization problems in machine learning.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    What problem formulations does <b>Cooper</b> support?
  </summary>
  <div style="margin-left: 20px;">
    <b>Cooper</b> supports the following formulations:
    <ul>
      <li>Lagrangian Formulation</li>
      <li>Augmented Lagrangian Formulation</li>
    </ul>
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    When should I pick any of these formulations?
  </summary>
  <div style="margin-left: 20px;">
    <b>Lagrangian Formulation</b> is a good choice when ...
    <br>
    <b>Augmented Lagrangian Formulation</b> is a good choice when ...
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    What is a good starting configuration for a Cooper optimizer (primal and dual)?
  </summary>
  <div style="margin-left: 20px;">
    For the dual optimizer, we recommend using SGD with a learning rate not too high to avoid overshoots and setting `maximize=True`.
    <br>
    For the primal optimizer, we recommend ...
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    Which <b>Cooper</b> optimizer should I use?
  </summary>
  <div style="margin-left: 20px;">
    <b>Cooper</b> provides a range of optimizers to choose from. The <b>AlternatingDualPrimalOptimizer</b> is a good starting point.
  </div>
</details>

### Debugging and troubleshooting

<details>
  <summary style="font-size: 1.2rem;">
    Why is my problem not becoming feasible?
  </summary>
  <div style="margin-left: 20px;">
    Answer here.
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
    Answer here.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    What quantities should I log for sanity-checking?
  </summary>
  <div style="margin-left: 20px;">
    Answer here.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    What do typical multiplier dynamics look like?
  </summary>
  <div style="margin-left: 20px;">
    Answer here.
  </div>
</details>

<details>
  <summary style="font-size: 1.2rem;">
    What should I do if my Lagrange multipliers diverge?
  </summary>
  <div style="margin-left: 20px;">
    Answer here.
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

If non convex
Or stochastic
Autograd differentiable objective and constraints (or non-differentiable constraints but with a surrogate)
Something about CMPState data structure
Argue for cheap cost (for free, compared to general minmax game)
Gradient of primal Lagrangian is autograd-friendly
Gradient of a linear combination of functions
Why are they useful?
What should I do if they oscillate too much?
What if they don’t stabilize/converge?
Complementary slackness
Dynamics/Solution
Loss/Lagrangian/ConstraintViolation
