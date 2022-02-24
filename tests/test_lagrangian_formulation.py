#!/usr/bin/env python

"""Tests for Lagrangian Formulation class."""

import pytest
import testing_utils
import torch

import cooper


def test_lagrangian_formulation():
    class CustomCMP(cooper.ConstrainedMinimizationProblem):
        def __init__(self, is_constrained=False):
            super().__init__(is_constrained)

        def update_state(self, ineq_defect=None, eq_defect=None):
            self.ineq_defect = ineq_defect
            self.eq_defect = eq_defect

    cmp = CustomCMP(is_constrained=True)

    lf = cooper.LagrangianFormulation(cmp)
    cmp.update_state(eq_defect=torch.tensor([1.0]))
    lf.create_state()
    assert (lf.ineq_multipliers is None) and (lf.eq_multipliers is not None)

    lf = cooper.LagrangianFormulation(cmp)
    cmp.update_state(
        eq_defect=torch.tensor([1.0]), ineq_defect=torch.tensor([1.0, 1.2])
    )
    lf.create_state()
    assert (lf.ineq_multipliers is not None) and (lf.eq_multipliers is not None)
