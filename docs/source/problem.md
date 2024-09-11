```{eval-rst}
.. currentmodule:: cooper.problem
```

(cmp)=

# Constrained Minimization Problem

We consider constrained minimization problems (CMPs) expressed as:

$$
\min_{\mathbf{x} \in \Omega} & \,\, f(\mathbf{x}) \\ \text{s.t. }
& \,\, \mathbf{g}(\mathbf{x}) \le \mathbf{0} \\ & \,\, \mathbf{h}(\mathbf{x}) = \mathbf{0}
$$

Here $\Omega$ represents the domain of definition of the functions
$f, \mathbf{g}$ and $\mathbf{h}$. Note that $f$ is a scalar-valued function, whereas
$\mathbf{g}$ and $\mathbf{h}$ are vector-valued functions. We group together all the
inequality constraints in $\mathbf{g}$ and all the equality constraints in $\mathbf{h}$.
In other words, a component function $h_i(x)$ corresponds to the scalar constraint
$h_i(\mathbf{x}) = 0$.

:::{admonition} Brief notes on conventions and terminology

- We refer to $f$ as the **loss** or **objective** to be minimized.
- We adopt the convention $g(\mathbf{x}) \le 0$ for inequality constraints and
  $h(\mathbf{x}) = 0$ for equality constraints. If your constraints are different,
  for example $g(\mathbf{x}) \ge \epsilon$, you should provide **Cooper** with
  $\epsilon - g(\mathbf{x}) \le 0$.
- We use the term **constraint violation** to refer to $\mathbf{g}(\mathbf{x})$ and
    $\mathbf{h}(\mathbf{x})$.
  that equality constraints $h(x)$ are satisfied *only* when their
  defect is zero. On the other hand, a *negative* defect for an inequality
  constraint  $g(x)$ means that the constraint is *strictly* satisfied;
  while a *positive* defect means that the inequality constraint is being
  violated.
:::

## Constraints
TODO


## CMP State

We represent computationally the "state" of a CMP using a {py:class}`CMPState`
object. A `CMPState` is a {py:class}`dataclasses.dataclass` which contains the
information about the loss and equality/inequality violations at a given point
$x$. If a problem has no equality or inequality constraints, these
arguments can be omitted in the creation of the `CMPState`.

:::{admonition} Stochastic estimates in `CMPState`
:class: important

In problems for which computing the loss or constraints exactly is prohibitively
expensive, the {py:class}`CMPState` may contain stochastic estimates of the
loss/constraints. For example, this is the case when the loss corresponds to a
sum over a large number of terms, such as training examples. In this case, the
loss and constraints may be estimated using mini-batches of data.

Note that, just as in the unconstrained case, these approximations can
entail a compromise in the stability of the optimization process.
:::

```{eval-rst}
.. autoclass:: CMPState
    :members: as_tuple
```

For details on the use of proxy-constraints and the `proxy_ineq_defect` and
`proxy_eq_defect` attributes, please see {ref}`lagrangian_formulations`.

## Constrained Minimization Problem

```{eval-rst}
.. autoclass:: ConstrainedMinimizationProblem
    :members:
```

## Example

The example below illustrates the main steps that need to be carried out to
define a `ConstrainedMinimizationProblem` in **Cooper**.

1. *\[Line 4\]* Define a custom class which inherits from {py:class}`ConstrainedMinimizationProblem`.
2. *\[Line 10\]* Write a closure function that computes the loss and constraints.
3. *\[Line 14\]* Note how the `misc` attribute can be use to store previous results.
4. *\[Line 18\]* Return the information about the loss and constraints packaged into a {py:class}`CMPState`.
5. *\[Line 18\]* (Optional) Modularize the code to allow for evaluating the constraints `only`.

```{code-block} python
:emphasize-lines: 4,10,14,18,20
:linenos: true

import torch
import cooper

class MyCustomCMP(cooper.ConstrainedMinimizationProblem):
    def __init__(self, problem_attributes, criterion):
        self.problem_attributes = problem_attributes
        self.criterion = criterion
        super().__init__()

    def closure(self, model, inputs, targets):

        cmp_state = self.defect_fn(model, inputs, targets)

        logits = cmp_state.misc["logits"]
        loss = self.criterion(logits, targets)
        cmp_state.loss = loss

        return cmp_state

    def defect_fn(self, model, inputs, targets):

        logits = model.forward(inputs)

        const_level0, const_level1 = self.problem_attributes

        # Remember to write the constraints using the convention "g <= 0"!

        # (Greater than) Inequality that only depends on the model properties or parameters
        # g_0 >= const_level0 --> const_level0 - g_0 <= 0
        defect0 = const_level0 - ineq_const0(model)

        # (Less than) Inequality that depends on the model's predictions
        # g_1 <= const_level1 --> g_2  - const_level1 <= 0
        defect1 = ineq_const1(logits) - const_level1

        # We recommend using torch.stack to ensure the dependencies in the computational
        # graph are properly preserved.
        ineq_defect = torch.stack([defect0, defect1])

        return cooper.CMPState(ineq_defect=ineq_defect, eq_defect=None, misc={'logits': logits})
```

:::{warning}
**Cooper** is primarily oriented towards **non-convex** CMPs that arise
in many machine/deep learning settings. That is, problems for which one of
the functions $f, g, h$ or the set $\Omega$ is non-convex.

Whenever possible, we provide references to appropriate literature
describing convergence results for our implemented (under suitable
assumptions). In general, however, the use of Lagrangian-based approaches
for solving non-convex CMPs does not come with guarantees regarding
optimality or feasibility.

Some theoretical results can be obtained when considering mixed strategies
(distributions over actions for the primal and dual players), or by relaxing
the game-theoretic solution concept (i.e. aiming for approximate/correlated
equilibria), even for problems which are non-convex on the primal (model)
parameters. For more details, see the work of {cite:t}`cotter2019JMLR` and
{cite:t}`lin2020gradient` and references therein. We plan to include some
of these techniques in future versions of **Cooper**.

If you are dealing with optimization problems under "nicely behaved" convex
constraints (e.g. cones or $L_p$-balls) we encourage you to check out
[CHOP](https://github.com/openopt/chop). If your problems involves "manifold"
constraints (e.g. orthogonal or PSD matrices), you might consider using
[GeoTorch](https://github.com/Lezcano/geotorch).
:::

```{eval-rst}
.. currentmodule:: cooper.formulation
```

## Formulation

TODO: move somewhere else?

Formulations denote mathematical or algorithmic techniques aimed at solving a
specific (family of) CMP. **Cooper** is heavily (but not exclusively!) designed
for an easy integration of Lagrangian-based formulations. You can find more
details in {doc}`lagrangian_formulation`.

```{eval-rst}
.. autoclass:: Formulation
    :members:
```
