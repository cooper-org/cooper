(penalty_coefficients)=

# Penalty Coefficients

```{eval-rst}
.. currentmodule:: cooper.multipliers
```

Analogous to {ref}`multipliers`, ...

## Example


## Base Class

```{eval-rst}
.. autoclass:: PenaltyCoefficient
    :members:
```

## Dense Penalty Coefficients

```{eval-rst}
.. autoclass:: DensePenaltyCoefficient
    :members
```

## Indexed Penalty Coefficients

```{eval-rst}
.. autoclass:: IndexedPenaltyCoefficient
  :members
```

:::{note}
**Cooper** supports vector-valued penalty coefficients that match the size of a constraint. This can be done by passing a tensor of coefficients to the `init` argument of a {py:class}`~cooper.multipliers.PenaltyCoefficient`, where each element corresponds to a penalty coefficient for an individual constraint.
:::

Since it is often desirable to increase the penalty coefficient over the optimization process, **Cooper** provides a scheduler mechanism to do so. For more information, see [Penalty Coefficient Updaters](#penalty-coefficient-updaters).


## Penalty Coefficient Updaters
