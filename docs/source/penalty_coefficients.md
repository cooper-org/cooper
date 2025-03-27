(penalty_coefficients)=

# Penalty Coefficients

```{eval-rst}
.. currentmodule:: cooper.penalty_coefficients
```

{py:class}`~cooper.penalty_coefficients.PenaltyCoefficient` objects are used to represent the penalty coefficients $\vc_{\vg}$ and $\vc_{\vh}$, required by formulations such as the {py:class}`~cooper.formulations.QuadraticPenalty`
and {py:class}`~cooper.formulations.AugmentedLagrangian` formulations.

In **Cooper**, penalty coefficients are wrappers around a {py:class}`torch.Tensor`. Notably, they do not require gradients, as they are not optimized.

The `cooper.penalty_coefficients` module provides the following types of penalty coefficients:

- **{py:class}`~cooper.penalty_coefficients.DensePenaltyCoefficient`**: Models each penalty coefficient individually.
- **{py:class}`~cooper.penalty_coefficients.IndexedPenaltyCoefficient`**: Similar to {py:class}`~cooper.penalty_coefficients.DensePenaltyCoefficient`, but allows fetching and updating penalty coefficients by index. This is useful when constraints are sampled, so that the required penalty coefficients change at each iteration.

:::{admonition} Linking constraints and penalty coefficients
:class: hint

{py:class}`~cooper.constraints.Constraint` objects require an associated {py:class}`~cooper.penalty_coefficients.PenaltyCoefficient` object to be passed to the constructor when the problem formulation demands it. You can check this requirement using the {py:attr}`~cooper.formulations.Formulation.expects_penalty_coefficient` attribute of a {py:class}`~cooper.formulations.Formulation` subclass.

```python
penalty_coefficient = ...
constraint = cooper.Constraint(
    penalty_coefficient=penalty_coefficient,
    constraint_type=cooper.ConstraintType.INEQUALITY,
    formulation_type=cooper.formulations.QuadraticPenalty,
)
```
:::

:::{note}
The helper methods {py:meth}`CMP.penalty_coefficients<.ConstrainedMinimizationProblem.penalty_coefficients>` and {py:meth}`CMP.named_penalty_coefficients<.ConstrainedMinimizationProblem.named_penalty_coefficients>` allow iteration over the penalty coefficients associated with constraints registered in a {py:class}`CMP<cooper.cmp.ConstrainedMinimizationProblem>`.
For more details, see [Registering constraints in a CMP](#registering-constraints).
:::

Consider the following Quadratic Penalty formulation of a constrained optimization problem:

$$
\Lag^{\text{QP}}_{\vc_g, \vc_h}(\vx) = f(\vx) + \frac{1}{2} \vc_{\vg}^\top \, \texttt{relu}(\vg(\vx))^2 + \frac{1}{2} \vc_{\vh}^\top \, \vh(\vx)^2,
$$

where $\vc_{\vg}$ and $\vc_{\vh}$ are the penalty coefficients associated with the inequality and equality constraints, respectively.

In **Cooper**, {py:class}`~cooper.penalty_coefficients.PenaltyCoefficient` objects represent the vectors $\vc_{\vg}$ and $\vc_{\vh}$, with one coefficient for each constraint. Alternatively, **Cooper** also supports scalar-valued penalty coefficients, which apply a shared coefficient across all constraints (i.e., $\vc_{\vg} = c_g \mathbf{1}$ and $\vc_{\vh} = c_h \mathbf{1}$).

Since it is often desirable to increase the penalty coefficient over the optimization process, **Cooper** provides a scheduler mechanism to do so. For more information, see [Penalty Coefficient Updaters](#penalty-coefficient-updaters).

### Initialization

To initialize a {py:class}`~cooper.penalty_coefficients.PenaltyCoefficient`, you can pass either a {py:class}`torch.Tensor` of shape `(num_constraints, )` or a scalar tensor to the `init` argument.

```python
penalty_coefficient = cooper.penalty_coefficients.DensePenaltyCoefficient(
    init=torch.ones(10, device="cuda", dtype=torch.float32)
)

penalty_coefficient = cooper.penalty_coefficients.IndexedPenaltyCoefficient(
    init=torch.tensor(1.0, device="cuda", dtype=torch.float32)
)
```

### Evaluating a {py:class}`~cooper.penalty_coefficients.PenaltyCoefficient`

Similar to [multipliers](multipliers.md), penalty coefficients can be evaluated using {py:meth}`~cooper.penalty_coefficients.PenaltyCoefficient.__call__`. For example:

```python
# `DensePenaltyCoefficient`s do not require arguments during evaluation
penalty_coefficient_value = penalty_coefficient()

# `IndexedPenaltyCoefficient`s require indices for evaluation
indices = torch.tensor([1, 2, 4, 6])
penalty_coefficient_value = penalty_coefficient(indices)
```

```{eval-rst}
.. autoclass:: PenaltyCoefficient
    :members: __call__, sanity_check, value, to, state_dict, load_state_dict
```

## Dense Penalty Coefficients

The {py:class}`~cooper.penalty_coefficients.DensePenaltyCoefficient` class wraps  a tensor of penalty coefficients, ensuring that all coefficients are accessed during each evaluation.

```{eval-rst}
.. autoclass:: DensePenaltyCoefficient
    :members: __call__
```

## Indexed Penalty Coefficients

Similar to {py:class}`~cooper.multipliers.IndexedMultiplier`s, {py:class}`~cooper.penalty_coefficients.IndexedPenaltyCoefficient`s allow fetching and updating the penalty coefficients *by index*. Given indices `idx`, the {py:meth}`~cooper.penalty_coefficients.IndexedPenaltyCoefficient.__call__()` method of an {py:class}`~cooper.penalty_coefficients.IndexedPenaltyCoefficient` object returns the penalty coefficients corresponding to the
indices in `idx`.

```{eval-rst}
.. autoclass:: IndexedPenaltyCoefficient
    :members: __call__
```

## Checkpointing

To save the current penalty coefficients of a {py:class}`CMP<cooper.cmp.ConstrainedMinimizationProblem>`, use the {py:meth}`~cooper.ConstrainedMinimizationProblem.state_dict()` method to create a state checkpoint. Later, you can restore this state using {py:meth}`~cooper.ConstrainedMinimizationProblem.load_state_dict()`. This process captures the multiplier and penalty coefficient values (see [CMP Checkpointing](#cmp-checkpointing) for details).

## Penalty Coefficient Updaters

Penalty coefficient updaters are used to adjust penalty coefficients during optimization based on constraint violations.
Cooper provides two feasibility-driven updaters: {py:class}`~cooper.penalty_coefficients.MultiplicativePenaltyCoefficientUpdater`
and {py:class}`~cooper.penalty_coefficients.AdditivePenaltyCoefficientUpdater`, both of which are driven by feasibility conditions.

```{eval-rst}
.. autoclass:: PenaltyCoefficientUpdater
    :members: step
```

### Multiplicative Penalty Coefficient Updater

The {py:class}`~cooper.penalty_coefficients.MultiplicativePenaltyCoefficientUpdater` multiplies the penalty coefficient
by a growth factor when `violation`s are above the tolerance.

```{eval-rst}
.. autoclass:: MultiplicativePenaltyCoefficientUpdater
```

### Additive Penalty Coefficient Updater

The {py:class}`~cooper.penalty_coefficients.AdditivePenaltyCoefficientUpdater` increases the penalty coefficient by a
fixed increment when `violation`s exceed the tolerance.

```{eval-rst}
.. autoclass:: AdditivePenaltyCoefficientUpdater
```

### Using Penalty Updaters in Training

To use a penalty coefficient updater in training:

1. **Instantiate** the updater with desired parameters.
2. **Call** {py:meth}`~cooper.penalty_coefficients.PenaltyCoefficientUpdater.step` with observed constraints after each optimization step.

**Example**:

```python
penalty_updater = cooper.penalty_coefficients.MultiplicativePenaltyCoefficientUpdater(
    growth_factor=2.0,
    violation_tolerance=1e-3,
    has_restart=True
)
roll_out = cooper_optimizer.roll(...)
penalty_updater.step(roll_out.cmp_state.observed_constraints)
```
