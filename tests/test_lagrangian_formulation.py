#!/usr/bin/env python

"""Tests for Lagrangian Formulation class."""

import pytest
import testing_utils
import torch

import cooper


def test_lagrangian_formulation():

    cmp = cooper.ConstrainedMinimizationProblem(is_constrained=True)

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
