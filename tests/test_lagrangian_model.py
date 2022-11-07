#!/usr/bin/env python

"""Tests for Lagrangian Formulation class."""

import torch

import cooper


def test_lagrangian_model():
    
    # create a model
    ineq_model = torch.nn.Linear(10, 1)
    eq_model = torch.nn.Linear(10, 1)
    # create a multiplier model
    ineq_multiplier_model = cooper.multipliers.MultiplierModel(ineq_model)
    eq_multiplier_model = cooper.multipliers.MultiplierModel(eq_model)
    lf = cooper.formulation.LagrangianModelFormulation(ineq_multiplier_model , eq_multiplier_model )
    # test lf state
    eq_featurs = torch.randn(100, 10)
    ineq_featurs = torch.randn(100, 10)
    ineq_state, eq_state = lf.state(ineq_featurs, eq_featurs)

    assert ineq_state is not None
    assert eq_state is not None

    # lf = cooper.LagrangianFormulation(cmp)
    # cmp.state = cooper.CMPState(
    #     eq_defect=torch.tensor([1.0]), ineq_defect=torch.tensor([1.0, 1.2])
    # )
    # lf.create_state(cmp.state)
    # assert (lf.ineq_multipliers is not None) and (lf.eq_multipliers is not None)
