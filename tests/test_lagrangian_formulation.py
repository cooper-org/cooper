#!/usr/bin/env python

"""Tests for Lagrangian Formulations."""

import torch

import cooper


def test_lagrangian_formulation():
    class DummyCMP(cooper.ConstrainedMinimizationProblem):
        def __init__(self):
            super().__init__(is_constrained=True)

        def closure(self):
            pass

    cmp = DummyCMP()

    lf = cooper.LagrangianFormulation(cmp)
    cmp.state = cooper.CMPState(eq_defect=torch.tensor([1.0]))
    lf.create_state(cmp.state)
    assert (lf.ineq_multipliers is None) and (lf.eq_multipliers is not None)

    lf = cooper.LagrangianFormulation(cmp)
    cmp.state = cooper.CMPState(
        eq_defect=torch.tensor([1.0]), ineq_defect=torch.tensor([1.0, 1.2])
    )
    lf.create_state(cmp.state)
    assert (lf.ineq_multipliers is not None) and (lf.eq_multipliers is not None)


def test_damped_lagrangian_formulation():
    class DummyCMP(cooper.ConstrainedMinimizationProblem):
        def __init__(self):
            super().__init__(is_constrained=True)

        def closure(self):
            pass

    cmp = DummyCMP()
    damping_coefficient = 10.0

    lf = cooper.DampedLagrangianFormulation(cmp, damping_coefficient)
    cmp.state = cooper.CMPState(eq_defect=torch.tensor([1.0]))
    lf.create_state(cmp.state)
    # Check that constraint multipliers are created correctly
    assert (lf.ineq_multipliers is None) and (lf.eq_multipliers is not None)

    # Check that the damping coefficient is set correctly
    lf = cooper.DampedLagrangianFormulation(cmp, damping_coefficient)
    cmp.state = cooper.CMPState(
        eq_defect=torch.tensor([1.0]), ineq_defect=torch.tensor([1.0, 1.2])
    )

    lf.create_state(cmp.state)
    assert (lf.ineq_multipliers is not None) and (lf.eq_multipliers is not None)
