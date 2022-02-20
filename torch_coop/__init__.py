"""Top-level package for Constrained Optimization in Pytorch."""

__author__ = """Jose Gallego-Posada"""
__email__ = "jgalle29@gmail.com"
__version__ = "0.1.0"

from torch_coop.constrained_optimizer import ConstrainedOptimizer
from torch_coop.problem import CMPState, ConstrainedMinimizationProblem
from torch_coop.lagrangian_formulation import LagrangianFormulation

from . import optim
