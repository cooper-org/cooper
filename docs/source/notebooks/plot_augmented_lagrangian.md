---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
---

# Using the Augmented Lagrangian function.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cooper-org/cooper/blob/master/docs/source/notebooks/plot_augmented_lagrangian.ipynb)

In this tutorial we demonstrate how to use the {py:class}`~cooper.formulations.AugmentedLagrangian` formulation to solve constrained optimization problems in **Cooper**. We borrow a simple 2D example from {cite:p}`nocedal2006NumericalOptimization` to illustrate the usage of the formulation, and its advantages over the {py:class}`~cooper.formulations.QuadraticPenalty` formulation.

```{code-cell} ipython3
%%capture
# %pip install cooper-optim
%pip install --index-url https://test.pypi.org/simple/ --no-deps cooper-optim  # TODO: Remove this line when cooper deployed to pypi
```

```{code-cell} ipython3

```

## Problem

+++

We consider solving the following problem (problem 17.3) from {cite:t}`nocedal2006NumericalOptimization`:

$$
\min_{\boldsymbol{x} \in \mathbb{R}^2} f(\boldsymbol{x}) = x_1 + x_2 \quad \text{s.t.} \quad x_1^2 + x_2^2 = 2.
$$

The Augmented Lagrangian function associated with this problem is:

$$
\mathcal{L}_{c}(\boldsymbol{x}, \mu) = x_1 + x_2 + \mu (x_1^2 + x_2^2 - 2) + \frac{c}{2} (x_1^2 + x_2^2 - 2)^2,
$$

where $\mu$ is the Lagrange multiplier associated with the equality constraint, and $c$ is the penalty parameter.

We will also consider the Quadratic Penalty function associated with this problem: TODO
