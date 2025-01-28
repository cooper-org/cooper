(penalty_coefficients)=

# Penalty Coefficients

```{eval-rst}
.. currentmodule:: cooper.penalty_coefficients
```

Penalty coefficient objects represent the penalty coefficients $\vc_{\vg}$ and $\vc_{\vh}$, required by formulations such as the {py:class}`~cooper.formulations.QuadraticPenalty`
and {py:class}`~cooper.formulations.AugmentedLagrangian` formulations.

In **Cooper**, penalty coefficients are wrappers around {py:class}`torch.Tensor`.

The `cooper.penalty_coefficients` module provides the following types of penalty coefficients:
- **{py:class}`~cooper.penalty_coefficients.DensePenaltyCoefficient`**: Models each penalty coefficient individually.
- **{py:class}`~cooper.penalty_coefficients.IndexedPenaltyCoefficient`**: Similar to {py:class}`~cooper.penalty_coefficients.DensePenaltyCoefficient` but allows fetching and updating penalty coefficients by index. Useful when constraints are sampled, and thus the required penalty coefficients change at each iteration.

## Dense Penalty Coefficients

{py:class}`~cooper.penalty_coefficients.DensePenaltyCoefficient` objects model each penalty coefficient individually.
Incase of a single penalty coefficient, a scalar can be passed to the `init` argument.
For multiple penalty coefficients, a `(num_constraints, )` shape {py:class}`torch.Tensor` can be passed.

```{eval-rst}
.. autoclass:: DensePenaltyCoefficient
    :members: __call__
```

To initialize a {py:class}`~cooper.penalty_coefficients.DensePenaltyCoefficient`, you can pass a scalar or a `(num_constraints, )`
shape {py:class}`torch.Tensor` to the `init` argument.

```python
penalty_coefficient = cooper.penalty_coefficients.DensePenaltyCoefficient(
    init=torch.tensor(1.0)
)
```

## Indexed Penalty Coefficients

{py:class}`~cooper.penalty_coefficients.IndexedPenaltyCoefficient` objects allow fetching and updating the penalty coefficients
*by index*. Given indices `idx`, the {py:meth}`~cooper.penalty_coefficients.IndexedPenaltyCoefficient.__call__()` method of
an {py:class}`~cooper.penalty_coefficients.IndexedPenaltyCoefficient` object returns the penalty coefficients corresponding to the
indices in `idx`.
{py:class}`~cooper.penalty_coefficients.IndexedPenaltyCoefficient` objects are designed for situations where only a subset of constraints are observed at each iteration, rather than all constraints.
This approach is especially useful when the number of constraints is large, such as in tasks where a constraint is imposed for each data point. In these cases, measuring all constraints at once can be computationally prohibitive.

```{eval-rst}
.. autoclass:: IndexedPenaltyCoefficient
    :members: __call__
```

```python
penalty_coefficient = cooper.penalty_coefficients.IndexedPenaltyCoefficient(
    init=torch.tensor(1.0)
)
```

## Base Class

```{eval-rst}
.. autoclass:: PenaltyCoefficient
    :members: to, state_dict, load_state_dict
```

:::{note}
**Cooper** supports vector-valued penalty coefficients that match the size of a constraint. This can be done by passing a tensor of coefficients to the `init` argument of a {py:class}`~cooper.penalty_coefficients.PenaltyCoefficient`, where each element corresponds to a penalty coefficient for an individual constraint.
:::

Since it is often desirable to increase the penalty coefficient over the optimization process, **Cooper** provides a scheduler mechanism to do so. For more information, see [Penalty Coefficient Updaters](#penalty-coefficient-updaters).

## Penalty Coefficient Updaters

Penalty coefficient updaters are objects that update the penalty coefficients of a {py:class}`~cooper.penalty_coefficients.PenaltyCoefficient` object.

```{eval-rst}
.. autoclass:: PenaltyCoefficientUpdater
    :members: __call__
```

```{eval-rst}
.. autoclass:: MultiplicativePenaltyCoefficientUpdater
    :members: __call__
```
