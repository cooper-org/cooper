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

    def forward(self):
        w = self.weight
        if self.positive:
            w.data = torch.relu(w).data
        return w

    def __str__(self):
        return str(self.forward().item())
