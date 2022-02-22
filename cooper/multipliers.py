#!/usr/bin/env python

"""Class for Lagrange multipliers."""

import torch


class DenseMultiplier(torch.nn.Module):
    def __init__(self, init, *, positive=False):
        super().__init__()
        self.weight = torch.nn.Parameter(init)
        self.positive = positive

    @property
    def shape(self):
        return self.weight.shape

    @property
    def grad(self):
        return self.weight.grad

    def forward(self):
        return self.weight

    def project_(self):
        # Generic projection for non-negative multipliers used in inequality
        # constraints. May be generalized to other custom projections
        if self.positive:
            self.weight.data = torch.relu(self.weight).data

    def __str__(self):
        return str(self.forward().data)

    def __repr__(self):
        pos_str = "inequality" if self.positive else "equality"
        rep = "DenseMultiplier(" + pos_str + ", " + str(self.weight.data) + ")"
        return rep
