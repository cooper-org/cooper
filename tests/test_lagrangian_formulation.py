#!/usr/bin/env python

"""Tests for Lagrangian Formulation class."""

import torch
import torch_coop

import pytest
import testing_utils


def test_lagrangian_formulation():

    cmp = torch_coop.ConstrainedMinimizationProblem(is_constrained=True)
    lf = torch_coop._LagrangianFormulation(cmp)

    lf.create_state(eq_defect=torch.tensor([1.0]))
    assert (lf.ineq_multipliers is None) and (lf.eq_multipliers is not None)

    lf.create_state(ineq_defect=torch.tensor([1.0, 1.2]))
    assert (lf.ineq_multipliers is not None) and (lf.eq_multipliers is not None)
